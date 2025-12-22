import jax
import jax.numpy as jnp

def split_heads(x, num_heads):
    batch, seq_len, hidden_dim = x.shape
    head_dim = hidden_dim // num_heads
    x = x.reshape(batch, seq_len, num_heads, head_dim)
    return jnp.transpose(x, (0, 2, 1, 3)) 

def merge_heads(x):
    batch, num_heads, seq_len, head_dim = x.shape
    x = jnp.transpose(x, (0, 2, 1, 3))
    return x.reshape(batch, seq_len, num_heads * head_dim)

def multi_head_attention(params, X, num_heads, *, causal=True):
    """
    X: [batch, seq_len, hidden_dim]
    params: (W_q, W_k, W_v, W_o)
    """
    batch, seq_len, hidden_dim = X.shape
    head_dim = hidden_dim // num_heads

    W_q, W_k, W_v, W_o = params

    Q = X @ W_q
    K = X @ W_k
    V = X @ W_v

    Q = split_heads(Q, num_heads)
    K = split_heads(K, num_heads)
    V = split_heads(V, num_heads)

    scores = (Q @ jnp.swapaxes(K, -1, -2)) / jnp.sqrt(head_dim)
    scores = scores.astype(jnp.float32)

    if causal:
        mask = jnp.tril(jnp.ones((seq_len, seq_len), dtype=jnp.float32))
        mask = mask[None, None, :, :]
        scores = scores - 1e10 * (1.0 - mask)

    weights = jax.nn.softmax(scores, axis=-1)
    attended = weights @ V

    merged = merge_heads(attended)
    out = merged @ W_o
    return out
