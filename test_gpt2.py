import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer

# Load the fine-tuned model and tokenizer
model_path = "./fine_tuned_gpt2"
model = GPT2LMHeadModel.from_pretrained(model_path)
tokenizer = GPT2Tokenizer.from_pretrained(model_path)

# Keep model on CPU to avoid MPS issues
device = torch.device("cpu")
model = model.to(device)
print("Using CPU for inference to avoid MPS compatibility issues.")

# Function to generate text
def generate_text(prompt, max_length=200):
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
    attention_mask = torch.ones_like(input_ids).to(device)
    output = model.generate(
        input_ids,
        attention_mask=attention_mask,
        max_length=max_length,
        num_return_sequences=1,
        temperature=0.7,
        do_sample=True,
        pad_token_id=tokenizer.eos_token_id
    )
    return tokenizer.decode(output[0], skip_special_tokens=True)

# Test prompts
test_prompts = [
    "Problem: def calculate_average_price(prices):",
    "Problem: Implement a binary search algorithm.",
    "Problem: Create a function to reverse a string.",
    "Problem: Write a function to find the maximum element in a list.",
    "Problem: Calculate the average price of a list of fashion items."
]

# Generate and print results
print("Testing the fine-tuned model:")
for prompt in test_prompts:
    print(f"\nPrompt: {prompt}")
    generated_text = generate_text(prompt)
    print(f"Generated text:\n{generated_text}\n")
    print("-" * 50)

print("Testing complete.")