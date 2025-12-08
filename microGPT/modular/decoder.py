from .attention_layer import MhAttention
import hax
import jax.numpy as jnp
import jax

class Dense(hax.Module):
    def __init__(self, units, activation=None):
        super().__init__()
        self.units = units
        self.activation = activation
    
    def call(self, x):
        _in = x.shape[-1]
        w = hax.param(
            self, 
            'weight', 
            shape=(_in, self.units), 
            init_fn= hax.initializers.VarianceScaling(2.0, 'fan_in', 'uniform')
            )
        b = hax.param(
            self, 
            'bias', 
            shape=(self.units,), 
            init_fn= hax.initializers.Constant(0.0)
            )
        out = jnp.matmul(x, w) + b

        if self.activation is not None:
            fun = getattr(jax.nn, self.activation)
            out = fun(out)
        return out
    
class FeedForward(hax.Module):
    def __init__(self, scale=2):
        super().__init__()
        self.scale = scale

    def call(self, x):
        _in = x.shape[-1]
        x = Dense(_in*self.scale, activation='gelu')(x)
        x = Dense(_in)(x)
        return x

def layer_norm(params, x, eps=1e-5):
    gamma, beta = params
    mean = jnp.mean(x, axis=-1, keepdims=True)
    std = jnp.std(x, axis=-1, keepdims=True)
    norm = (x - mean) / (std + eps)
    return gamma * norm + beta


class LayerNorm(hax.Module):
    def __init__(self):
        super().__init__()
        
    def call(self, x, eps=1e-5):
        mean = jnp.mean(x, axis=-1, keepdims=True)
        std = jnp.std(x, axis=-1, keepdims=True)
        norm = (x - mean) / (std + eps)

        gamma_init = hax.initializers.Constant(1.0)
        gamma = hax.param(self, 'gamma', shape=(x.shape[-1],), init_fn=gamma_init)

        beta_init = hax.initializers.Constant(0.0)
        beta = hax.param(self, 'beta', shape=(x.shape[-1],), init_fn=beta_init)

        return gamma * norm + beta
    

class GPTStyleBlock(hax.Module):
    def __init__(self, heads):
        super().__init__()
        self.heads = heads

    def call(self, x):
        ln_x = LayerNorm()(x) # pre-layernorm before attention
        attn_x = MhAttention(self.heads)(ln_x) # attention
        res_x = x + attn_x # residual

        ln_x = LayerNorm()(res_x) # pre-layernorm before ffn
        ffn_x = FeedForward()(ln_x) # feed-forward
        out = res_x + ffn_x # residual

        return out

class GPTStyleTransformer(hax.Module):
    def __init__(self, heads, num_layers):
        super().__init__()
        self.heads = heads
        self.num_layers = num_layers

    def call(self, x):
        layers = hax.ModuleStack(
            GPTStyleBlock(self.heads), 
            self.num_layers
            )
        return hax.Sequential(layers)(x)
