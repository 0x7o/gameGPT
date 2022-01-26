# gameGPT
[![use.py Test](https://github.com/0x7o/gameGPT/actions/workflows/use_test.yml/badge.svg)](https://github.com/0x7o/gameGPT/actions/workflows/use_test.yml)

[{habr article}](https://habr.com/ru/post/599715/)[{site}](https://0x7o.link/gamegpt/)

Trained ruGPT3 on Russian "Choose Your Own Adventure" stories

Version  | Sise |
--- | --- |
[gameGPT-small](https://huggingface.co/0x7194633/gameGPT-small) | 526 MB | 
[gameGPT-medium](https://huggingface.co/0x7194633/gameGPT-medium) | 1.42 GB | 
[gameGPT-large](https://huggingface.co/0x7194633/gameGPT-large) | 2.93 GB | 

# Usage
Example usage:

[![Try Model Training In Colab!](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/0x7o/text2keywords/blob/main/example/keyT5_use.ipynb)

```
pip install transformers
```

```python
from transformers import GPT2LMHeadModel, GPT2Tokenizer

model_name = '0x7194633/gameGPT-medium'
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
model = GPT2LMHeadModel.from_pretrained(model_name, pad_token_id=tokenizer.eos_token_id)

# Install a Nvidia gpu if necessary
#model.cuda()

def generate(text, **kwargs):
    input_ids = tokenizer.encode(text, return_tensors='pt')
    out = model.generate(input_ids, **kwargs)
    return tokenizer.decode(out[0])
    
act = '> Где я нахожусь?'
print(generate(act, max_length=100, top_p=0.7, temperature=1.0))
```
