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
