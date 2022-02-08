# gameGPT
[![use.py Test](https://github.com/0x7o/gameGPT/actions/workflows/use_test.yml/badge.svg)](https://github.com/0x7o/gameGPT/actions/workflows/use_test.yml)

Trained GPT2-XL on Choose Your Own Adventure stories

[The original, Tensorflow version](https://github.com/Latitude-Archives/AIDungeon)

[{ DEMO-GAME }](https://gamio.ru)

Version  | Loss | Perplexity |
--- | --- | --- |
[gameGPT](https://huggingface.co/0x7194633/gameGPT) | 0.75 | 2.2 |

# Usage
Example usage:

```
pip install torch transformers
```

```python
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer

model_name = '0x7194633/gameGPT'
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
model = GPT2LMHeadModel.from_pretrained(model_name)

if torch.cuda.is_available():
  model.cuda()
  
def generate(text, **kwargs):
  inpt = tokenizer.encode(text, return_tensors="pt")
  if torch.cuda.is_available():
    out = model.generate(inpt.cuda(), **kwargs)
  else:
    out = model.generate(inpt, **kwargs)
  return tokenizer.decode(out[0])
  

act = "Test"
print(generate(act, max_length=5, repetition_penalty=5.0, top_k=5, top_p=0.95, temperature=0.9))
```
