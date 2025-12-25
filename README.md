# microGPT 
An academic implementation of GPT: only math and JAX 

---

<img src="media/image-4.png" alt="transformer_diagram" width="100%">

**microGPT** is a reflection of how the original **Transformer** layers were engineered back in 2017 at **Google**. This is a **very low-level implementation** of GPT, built entirely from **math equations and JAX**.
Core components like **Self-Attention**, **Embeddings**, **LayerNorm**, and **Feedforward Networks** are implemented **from scratch**, designed to help newcomers understand the inner workings of **LLMs** — without hiding behind prebuilt abstractions.


## Setup
### Installation:
- clone the repo
- go to the project folder
- install

```bash
git clone https://github.com/kandarpa02/microGPT.git
cd microGPT
pip install .
pip install -r requirements.txt
```

## Inference
I added a simple generate function for inference, just import it and all weights and tokenizers will be downloaded automatically!

```python
from microGPT.infer import generate_text

import pprint
rng = jax.random.PRNGKey(443)
prompt = "Once upon a time"

text=generate_text(
        rng,
        prompt
    )

pprint.pprint(
    f"Result: {text}"
)

# ('Result: Once upon a time I was right on a good weather, that feels like '
#  'automatic useless witnesses may simply argue that, in a shameless lagon '
#  'study in a heart of photographer at me, before the trio-neaster was')

```