import numpy as np
import torch
import random
from transformers import GPT2LMHeadModel, GPT2Tokenizer

model_name = '0x7194633/gameGPT-large'
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
