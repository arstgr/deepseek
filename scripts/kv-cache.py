import time
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer

tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
model = GPT2LMHeadModel.from_pretrained("gpt2")

prompt = "The next day is bright"
inputs = tokenizer(prompt, return_tensors="pt")
input_ids = inputs.input_ids
attention_mask = inputs.attention_mask

# Timing without KV cache
start_time_without_cache = time.time()
output_without_cache = model.generate(input_ids,
    max_new_tokens=200,
    use_cache=False,
    attention_mask=attention_mask)
end_time_without_cache = time.time()
duration_without_cache = end_time_without_cache - start_time_without_cache
print(f"Time without KV Cache: {duration_without_cache:.4f} seconds")

#Timing with KV Cache
start_time_with_cache = time.time()
output_with_cache = model.generate(input_ids,
    max_new_tokens=200,
    use_cache=True,
    attention_mask=attention_mask)
end_time_with_cache = time.time()
duration_with_cache = end_time_with_cache - start_time_with_cache
print(f"Time with KV Cache: {duration_with_cache:.4f} seconds")

#Speedup
speedup = duration_without_cache/duration_with_cache
print(f"\nKV Cache Speedup: {speedup:.2f}x")
