import os
import sys
import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel, Trainer, TrainingArguments, DataCollatorForLanguageModeling
from datasets import load_dataset

def load_dataset_directly(dataset_name, split):
    print(f"Attempting to load dataset '{dataset_name}' directly...")
    try:
        dataset = load_dataset(
            dataset_name,
            split=split,
            download_mode="force_redownload",
        )
        print("Dataset loaded successfully.")
        return dataset
    except Exception as e:
        print(f"Error loading dataset: {e}")
        sys.exit(1)

# Load the dataset
dataset = load_dataset_directly("jinaai/code_exercises", "train")

# Use a small subset of the dataset
dataset = dataset.select(range(10000))  # Use only the first 1,000 examples

print(f"Dataset size: {len(dataset)} examples")
print(f"First example: {dataset[0]}")
print(f"Dataset features: {dataset.features}")

# Load the GPT-2 tokenizer and model (using the smallest GPT-2 variant)
print("Loading GPT-2 tokenizer and model...")
model_name = "distilgpt2"  
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
model = GPT2LMHeadModel.from_pretrained(model_name)

# Set the padding token
tokenizer.pad_token = tokenizer.eos_token

# Preprocess the dataset: tokenize the text
def preprocess_function(examples):
    texts = [f"Problem: {prob}\nSolution: {sol}" for prob, sol in zip(examples['problem'], examples['solution'])]
    return tokenizer(texts, truncation=True, max_length=128)  # Reduced max_length

print("Tokenizing the dataset...")
tokenized_datasets = dataset.map(preprocess_function, batched=True, remove_columns=dataset.column_names)

# Define data collator
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

# Define training arguments
training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="no",  # Disable evaluation
    learning_rate=2e-5,
    per_device_train_batch_size=2,  # Reduced batch size
    gradient_accumulation_steps=8,  # Increased gradient accumulation
    num_train_epochs=10,
    weight_decay=0.01,
    save_strategy="epoch",
    save_total_limit=2,
    logging_steps=50,
    report_to=["none"],  # Disable wandb logging
)

# Initialize the Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets,
    data_collator=data_collator,
    tokenizer=tokenizer,
)

# Train the model
print("Starting model training...")
trainer.train()

# Save the model
print("Saving the fine-tuned model...")
model.save_pretrained("./fine_tuned_gpt2")
tokenizer.save_pretrained("./fine_tuned_gpt2")

print("Fine-tuning complete. Model saved to ./fine_tuned_gpt2")

# Optional: Move model to MPS device for inference
if torch.backends.mps.is_available():
    model.to(torch.device("mps"))
    print("Model moved to MPS device for inference.")
else:
    print("MPS device not available. Model remains on CPU.")