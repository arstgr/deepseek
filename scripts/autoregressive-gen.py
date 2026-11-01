import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer

tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
model = GPT2LMHeadModel.from_pretrained("gpt2")

prompt = "The next day is bright"
inputs = tokenizer(prompt, return_tensors="pt")
input_ids = inputs.input_ids

print(f"Prompt: '{prompt}'", end="")

for _ in range(20):
    outputs = model(input_ids)
    logits = outputs.logits

    next_token_logits = logits[:, -1, :]
    next_token_id = next_token_logits.argmax(dim=-1).unsqueeze(-1)

    input_ids = torch.cat([input_ids, next_token_id], dim=-1)

    new_token = tokenizer.decode(next_token_id[0])
    print(new_token, end="", flush=True)

print("\n")
