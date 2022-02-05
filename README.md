# gameGPT
[![use.py Test](https://github.com/0x7o/gameGPT/actions/workflows/use_test.yml/badge.svg)](https://github.com/0x7o/gameGPT/actions/workflows/use_test.yml)

Trained ruGPT3 on Russian "Choose Your Own Adventure" stories

[{ DEMO-GAME }](https://gamio.ru)

Version  | Sise | Loss | Perplexity |
--- | --- | --- | --- |
[gameGPT](https://huggingface.co/0x7194633/gameGPT) | 670 MB | 2.15 | 5.2 |

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
  

act = "Тест"
print(generate(act, max_length=500, repetition_penalty=5.0, top_k=5, top_p=0.95, temperature=0.9))
```
