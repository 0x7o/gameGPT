![Cover](https://0x7o.link/gameGPT/cover.png "Cover")
# gameGPT

Trained GPT-J, GPT-Neo, GPT2-XL on Choose Your Own Adventure stories

[{ DEMO-GAME }](https://gamio.ru)

[gameGPT](https://huggingface.co/0x7194633/gameGPT)

[gameGPT-6B](https://huggingface.co/0x7194633/gameGPT-6B)

# Usage
gameGPT usage:

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

gameGPT-6B usage:

```
pip install torch transformers
```

```python
import torch
from from transformers import AutoTokenizer, AutoModelForCausalLM

model_name = '0x7194633/gameGPT'
tokenizer = AutoModelForCausalLM.from_pretrained(model_name)
model = AutoTokenizer.from_pretrained("EleutherAI/gpt-j-6B")

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
