# GPT-2 Fine-Tuning for Python Code Exercises

This repository contains code for fine-tuning a GPT-2 model to solve Python code exercises. It uses a dataset of coding problems and solutions and includes two main scripts: `fine_tune_gpt2.py` for model training and `test_gpt2.py` for testing the fine-tuned model.

## Repository Contents

- `fine_tune_gpt2.py`: Script to fine-tune a GPT-2 model on Python code exercises using the dataset `jinaai/code_exercises`. The model is trained to generate solutions for coding problems.
- `test_gpt2.py`: Script to test the fine-tuned GPT-2 model on new coding problems and evaluate its performance.

## Requirements

Install the required packages listed in `requirements.txt`:

##bash
`pip install -r requirements.txt`

## Main Libraries
transformers: Hugging Face's library for loading and fine-tuning language models.
datasets: Hugging Face's library for accessing and processing datasets.
torch: PyTorch, the deep learning framework used for model training.
wandb: For logging and tracking experiments (can be disabled).

##Fine-Tuning the GPT-2 Model
The fine_tune_gpt2.py script fine-tunes a pre-trained GPT-2 model (specifically distilgpt2) on a dataset of Python coding problems and solutions. The model is trained for 10 epochs with gradient accumulation and weight decay.

Steps:
Load the dataset using Hugging Face's datasets library.
Preprocess the dataset by tokenizing the problem and solution pairs.
Fine-tune the GPT-2 model using the Trainer API from the transformers library.
Save the fine-tuned model and tokenizer for future inference.
To run the script, use the following command:

bash
Copy code
`python fine_tune_gpt2.py`
Testing the Model
You can use the test_gpt2.py script to generate predictions from the fine-tuned model for new coding problems.

bash
Copy code
`python test_gpt2.py`
Example Usage
After fine-tuning, the model can generate Python solutions given a problem description. Below is a sample input and output:

Input: Problem: Write a function that returns the square of a number.
Output: Solution: def square(x): return x ** 2
Saving and Loading the Model
The fine-tuned model and tokenizer are saved to the ./fine_tuned_gpt2 directory. You can load them using the following code:

```python Copy code
from transformers import GPT2LMHeadModel, GPT2Tokenizer

tokenizer = GPT2Tokenizer.from_pretrained('./fine_tuned_gpt2')
model = GPT2LMHeadModel.from_pretrained('./fine_tuned_gpt2')

