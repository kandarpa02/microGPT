from microGPT.decoder.attn.attention import *
from microGPT.decoder.embed.embedding import *
from microGPT.decoder.layernorm.lnorm import *
from microGPT.decoder.linear.linear_layer import *
from microGPT.decoder.params.param_setup import *
import jax

def transformer_block(
    x,
    params,
    num_heads,
    *,
    rng=None,
    train=True,
    dropout_rate=0.1,
):
    """
    Pre-LN Transformer block
    params:
      {
        "ln1": (gamma, beta),
        "attn": (Wq, Wk, Wv, Wo),
        "ln2": (gamma, beta),
        "ffn_fc": (W, b),
        "ffn_proj": (W, b),
      }
    """

    # ---- Attention ----
    h = layer_norm(params["ln1"], x)
    attn_out = multi_head_attention(
        params["attn"],
        h,
        num_heads,
        causal=True,
    )
    def dropout(x, rate, rng, train: bool):
        if not train or rate == 0.0:
            return x
        keep_prob = 1.0 - rate
        mask = jax.random.bernoulli(rng, keep_prob, x.shape)
        return x * mask / keep_prob


    if train and dropout_rate > 0.0:
        rng, sub = jax.random.split(rng)
        attn_out = dropout(attn_out, rate=dropout_rate, rng=sub, train=train)

    x = x + attn_out

    # ---- FFN ----
    h = layer_norm(params["ln2"], x)
    h = linear(params["ffn_fc"], h)
    h = jax.nn.gelu(h)
    h = linear(params["ffn_proj"], h)

    if train and dropout_rate > 0.0:
        rng, sub = jax.random.split(rng)
        h = dropout(h, rate=dropout_rate, rng=sub, train=train)

    x = x + h
    return x, rng


class MicroGPT:
    def __init__(self, vocab, d_model, n_layers, n_heads):
        self.vocab = vocab
        self.d_model = d_model
        self.n_layers = n_layers
        self.n_heads = n_heads

    @staticmethod
    def run_fn(X, params, *, rng=None, train=True):
        """
        X: [batch, seq_len]
        """

        x = word_embedding(params["params"]["embed"], X)

        for block in params["params"]["blocks"]:
            x, rng = transformer_block(
                x,
                block,
                num_heads=params["config"]["n_heads"],
                rng=rng,
                train=train,
                dropout_rate=params["config"]["dropout"],
            )

        # Weight tying
        emb = params["params"]["embed"]["embedding_table"]
        logits = jnp.einsum("bsm,vm->bsv", x, emb)
        return logits


def init_gpt_params(
    rng,
    vocab,
    d_model,
    n_layers,
    n_heads,
    dropout=0.1,
):
    keys = jax.random.split(rng, n_layers + 1)

    params = {
        "config": {
            "n_heads": n_heads,
            "dropout": dropout,
            "pre_ln": True,
            "tie_embeddings": True,
        },
        "params":{
        "embed": init_embedding_params(42, vocab, d_model),
        "blocks": [],
    }
    }

    for i in range(n_layers):
        params["params"]["blocks"].append({
            "ln1": init_layer_norm_params(d_model),
            "attn": init_attention_param(d_model),
            "ln2": init_layer_norm_params(d_model),
            "ffn_fc": init_linear_param(d_model, 4 * d_model),
            "ffn_proj": init_linear_param(4 * d_model, d_model),
        })

    return params
