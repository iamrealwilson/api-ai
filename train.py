import os
import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel, Trainer, TrainingArguments, DataCollatorForLanguageModeling
from datasets import load_dataset, load_from_disk

# Define the dataset name and data format
dataset_name = "your_dataset_name"  # Replace with your dataset name
data_format = "your_data_format"  # Replace with your data format (e.g., "json", "text")

# Define the model and tokenizer
model_name = "gpt2"
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
model = GPT2LMHeadModel.from_pretrained(model_name)

# Load the dataset
if os.path.exists(dataset_name):
    dataset = load_from_disk(dataset_name)
else:
    dataset = load_dataset(dataset_name, data_format)

# Preprocess the dataset
def preprocess_function(examples):
    return tokenizer(examples["text"], truncation=True, padding="max_length", max_length=512)

dataset = dataset.map(preprocess_function, batched=True)

# Define the training arguments
training_args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=3,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    save_steps=1000,
    evaluation_strategy="steps",
    logging_steps=100,
    learning_rate=2e-5,
    weight_decay=0.01,
    fp16=True,
)

# Define the data collator
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

# Create the trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset["train"],
    eval_dataset=dataset["validation"],
    data_collator=data_collator,
)

# Train the model
trainer.train()

# Save the fine-tuned model
trainer.save_model("./finetuned_gpt2")