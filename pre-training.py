from transformers import GPT2Config, GPT2LMHeadModel, GPT2TokenizerFast, Trainer, TrainingArguments, TextDataset, DataCollatorForLanguageModeling

# Step 1: Define config for small GPT-2 (e.g., 124M-like)
config = GPT2Config(
    vocab_size=50257,
    n_positions=512,
    n_ctx=512,
    n_embd=768,
    n_layer=12,
    n_head=12,
)

model = GPT2LMHeadModel(config)
tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")

# Step 2: Save tokenizer vocab (optional if using custom tokenizer)
tokenizer.save_pretrained("my_gpt2_tokenizer")

# Step 3: Load your text file for pretraining
def load_dataset(file_path, tokenizer):
    return TextDataset(
        tokenizer=tokenizer,
        file_path=file_path,
        block_size=128
    )

dataset = load_dataset("toy_corpus.txt", tokenizer)
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

# Step 4: Train the model
training_args = TrainingArguments(
    output_dir="./gpt2-from-scratch",
    overwrite_output_dir=True,
    per_device_train_batch_size=2,
    num_train_epochs=5,
    save_steps=500,
    save_total_limit=2,
    prediction_loss_only=True,
    logging_steps=100,
)

trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=dataset,
)

trainer.train()
model.save_pretrained("./gpt2-from-scratch")
