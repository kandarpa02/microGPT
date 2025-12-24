import gdown
import pickle
import sentencepiece as spm
from .stack.gpt_micro import MicroGPT, init_gpt_params
import numpy as np
import jax

w_link = "https://drive.google.com/file/d/19K7BUygdtc61KGQn0h9Kq2IQcx0hb-7S/view?usp=sharing"
t_link = "https://drive.google.com/file/d/1-6Aueh3i372VdpC0eHMcxCFBlOjkKsNd/view?usp=sharing"

gdown.download(
    w_link,
    output="weights.pkl",
    fuzzy=True
)

gdown.download(
    t_link,
    output="tokenizer.model",
    fuzzy=True
)

sp = spm.SentencePieceProcessor("tokenizer.model")


def load_checkpoint(path):
    with open(path, "rb") as f:
        ckpt = pickle.load(f)
    return ckpt


ckpt = load_checkpoint("weights.pkl")
params = ckpt['params']

vocab = sp.vocab_size()
d_model = 256
n_layers = 4
n_heads  = 4
dropout = 0.1

gpt = MicroGPT(
    vocab=vocab,
    d_model=d_model,
    n_layers=n_layers,
    n_heads=n_heads,
)

import jax
import jax.numpy as jnp

@jax.jit
def forward_logits(params, tokens, rng):
    """
    tokens: [batch, seq_len] int32
    returns: [batch, seq_len, vocab]
    """
    return gpt.run_fn(tokens, params, rng=rng, train=False)

def _sample_next_token(logits, rng, temperature=1.0):
    """
    logits: [batch, vocab]
    """
    if temperature == 0.0:
        return jnp.argmax(logits, axis=-1)

    logits = logits / temperature
    return jax.random.categorical(rng, logits, axis=-1)

sample_next_token = jax.jit(_sample_next_token, static_argnames='temperature')

def generate_step(tokens, params, rng, temperature=1.0):
    rng, sub = jax.random.split(rng)

    logits = forward_logits(params, tokens, rng)
    next_logits = logits[:, -1, :]

    next_token = sample_next_token(next_logits, sub, temperature)
    next_token = next_token[:, None]

    tokens = jnp.concatenate([tokens, next_token], axis=1)
    return tokens, rng

import time

def generate(
    params,
    prompt,
    max_new_tokens=50,
    temperature=1.0,
    rng=None,
):
    """
    prompt: list[int] or 1D array
    """
    tokens = jnp.array(prompt, dtype=jnp.int32)[None, :]  # [1, seq]
    
    for _ in range(max_new_tokens):
        tokens, rng = generate_step(tokens, params, rng, temperature)

    return tokens[0]

def generate_text(
    rng,
    text,
    max_new_tokens=50,
    temperature=1.0,
):
    """
    Generate text continuations from a prompt using a trained language model.

    Args:
        rng: Random number generator key used for stochastic sampling.
        text (str): Input prompt text.
        max_new_tokens (int, optional): Maximum number of new tokens to generate.
            Defaults to 50.
        temperature (float, optional): Sampling temperature. Higher values increase
            randomness, lower values make generation more deterministic.
            Defaults to 1.0.

    Returns:
        str: Generated text including the prompt and newly generated tokens.
    """
    prompt_ids = sp.encode(text)
    out_ids = generate(
        params,
        prompt_ids,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        rng=rng,
    )
    return sp.decode(out_ids.tolist())
