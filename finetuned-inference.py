model = GPT2LMHeadModel.from_pretrained("./gpt2-finetuned-headlines")
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

input_text = "<|startoftext|> Headline:"
inputs = tokenizer(input_text, return_tensors="pt")
outputs = model.generate(**inputs, max_length=50, do_sample=True, temperature=0.7)

print(tokenizer.decode(outputs[0]))
