from transformers import GPT2LMHeadModel, GPT2Tokenizer

model = GPT2LMHeadModel.from_pretrained("./gpt2-from-scratch")  # or "gpt2"
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

prompt = "India is a country where"
inputs = tokenizer(prompt, return_tensors="pt")
outputs = model.generate(**inputs, max_length=50, do_sample=True, top_k=50)

print(tokenizer.decode(outputs[0], skip_special_tokens=True))
