import hax
import jax.numpy as jnp
import jax

class AddMm(hax.Module):
    def call(self, x, w, bias=True):
        out = jnp.matmul(x, w)
        if bias:
            init = hax.initializers.Constant(0.0)
            b = self.add_param('bias', [out.shape[-1]], init)
            return out + b

        else: return out

class MhAttention(hax.Module):
    def __init__(self, heads) -> None:
        super().__init__()
        self.heads = heads

    def call(self, x):
        batch, seq_len, hidden_dim = x.shape
        head_dim = hidden_dim // self.heads

        def split_heads(x, num_heads):
            batch, seq_len, hidden_dim = x.shape
            head_dim = hidden_dim // num_heads
            x = x.reshape(batch, seq_len, num_heads, head_dim)
            return jnp.transpose(x, (0, 2, 1, 3)) 

        def merge_heads(x):
            batch, num_heads, seq_len, head_dim = x.shape
            x = jnp.transpose(x, (0, 2, 1, 3))
            return x.reshape(batch, seq_len, num_heads * head_dim)
        
        qkv_init = hax.initializers.VarianceScaling(2.0, distribution="uniform")
        out_init = hax.initializers.VarianceScaling(1.0)

        qw = hax.param(self, 'q_weight', shape=[hidden_dim, hidden_dim], init_fn=qkv_init)
        kw = hax.param(self, 'k_weight', shape=[hidden_dim, hidden_dim], init_fn=qkv_init)
        vw = hax.param(self, 'v_weight', shape=[hidden_dim, hidden_dim], init_fn=qkv_init)
        ow = hax.param(self, 'o_weight', shape=[hidden_dim, hidden_dim], init_fn=out_init)

        def dotprod(x, w, bias=None):
            out = jnp.matmul(x, w)
            if bias is not None:
                init = hax.initializers.Constant(0.0)
                b = hax.param(self, f"{bias}", [out.shape[-1]], init)
                return out + b
            return out
        
        _Q = dotprod(x, qw, 'q_bias')
        _K = dotprod(x, kw, 'k_bias')
        _V = dotprod(x, vw, 'v_bias')

        num_heads = self.heads
        Q = split_heads(_Q, num_heads)
        K = split_heads(_K, num_heads)
        V = split_heads(_V, num_heads)


        scores = Q @ jnp.swapaxes(K, -1, -2) / jnp.sqrt(head_dim)
        scores = scores.astype(jnp.float32)
        
        mask = jnp.tril(jnp.ones((1, 1, seq_len, seq_len), dtype=jnp.float32))
        scores = scores - 1e10 * (1.0 - mask)

        weights = jax.nn.softmax(scores, axis=-1)
        attended = weights @ V

        merged = merge_heads(attended)
        out = merged @ ow
        return out