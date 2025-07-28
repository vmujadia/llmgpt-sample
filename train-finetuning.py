#<|startoftext|> Headline: Apple releases new iPhone. <|endoftext|>
#<|startoftext|> Headline: India wins cricket series. <|endoftext|>


from transformers import GPT2Tokenizer, GPT2LMHeadModel, TextDataset, DataCollatorForLanguageModeling, Trainer, TrainingArguments

model = GPT2LMHeadModel.from_pretrained("gpt2")
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
tokenizer.add_special_tokens({'pad_token': '[PAD]'})  # for training
model.resize_token_embeddings(len(tokenizer))

train_dataset = TextDataset(
    tokenizer=tokenizer,
    file_path="headlines_train.txt",
    block_size=128
)

data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer, mlm=False
)

training_args = TrainingArguments(
    output_dir="./gpt2-finetuned-headlines",
    overwrite_output_dir=True,
    num_train_epochs=3,
    per_device_train_batch_size=4,
    save_steps=200,
    save_total_limit=2,
    logging_steps=100,
    evaluation_strategy="no"
)

trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=train_dataset
)

trainer.train()
model.save_pretrained("./gpt2-finetuned-headlines")
