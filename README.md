# microGPT 
An academic implementation of GPT: only math and JAX 

---

<img src="media/image-4.png" alt="transformer_diagram" width="100%">

**microGPT** is a minimal, low-level implementation of a GPT-style Transformer, inspired by the original **“Attention Is All You Need” (2017)** architecture.

The entire model is implemented **from scratch using JAX**, with no high-level deep-learning abstractions. Core components such as:

- Token & positional embeddings  
- Multi-head self-attention  
- Layer normalization  
- Feed-forward networks  
- Autoregressive sampling  

are written directly from their **mathematical definitions**.

The goal of this project is **educational and research-focused**: to expose how modern LLMs work internally, without hiding logic behind frameworks like `nn.Transformer` or `flax.linen`.

### Model Details
- Parameters: **5.4M**
- Dataset: **OpenWebText-10k (50MB subset)**
- Library: **JAX**
- Training: custom training loop with autoregressive loss

This repository is intended for learners, researchers, and engineers who want to understand GPT models at the **implementation level**, not just the API level.

## Setup

### Installation

Clone the repository and install dependencies:

```bash
git clone https://github.com/kandarpa02/microGPT.git
cd microGPT
pip install -r requirements.txt
pip install .
```

## Inference
A minimal inference utility is provided via generate_text. On first use, pretrained weights and the tokenizer are downloaded automatically.

```python
import jax
from microGPT.infer import generate_text

rng = jax.random.PRNGKey(443)
prompt = "Once upon a time"

text = generate_text(
    rng=rng,
    text=prompt,
)

print(f"Result: {text}")

```
Output
```txt
Result: Once upon a time I was right on a good weather, that feels like automatic
useless witnesses may simply argue that, in a shameless lagon study in a heart of
photographer at me, before the trio-neaster was

```
---

```md
## Notes
- Decoder-only autoregressive Transformer
- Uses causal masking for next-token prediction
- Designed for educational and research purposes
```