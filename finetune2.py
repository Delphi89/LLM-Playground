import requests
from bs4 import BeautifulSoup
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import json

# Load the tokenizer
model_name = 'meta-llama/Meta-Llama-3-8B'
token = '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'
tokenizer = AutoTokenizer.from_pretrained(model_name, token=token)

# Define the path to the quantized model
quantized_model_path = 'quantized_model.pth'

try:
    # Load the quantized model from the local file system
    quantized_model = AutoModelForCausalLM.from_pretrained(quantized_model_path, local_files_only=True)

    # Check if CUDA is available and set the device accordingly
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('\nUsing device:', device)

    # Move the model to the appropriate device
    quantized_model = quantized_model.to(device)

    # Scrape data from the website
    url = 'https://cnimslatina.ro/despre'
    response = requests.get(url)
    soup = BeautifulSoup(response.text, 'html.parser')

    # Extract text and save to a file
    data = soup.get_text()
    with open('data.txt', 'w', encoding='utf-8') as f:
        f.write(data)

    # Prepare the dataset
    dataset = load_dataset('text', data_files={'train': 'data.txt'})

    # Tokenize the dataset
    def tokenize_function(examples):
        return tokenizer(examples['text'], truncation=True, padding='max_length')

    tokenized_datasets = dataset.map(tokenize_function, batched=True)

    # Inference with the quantized model
    inputs = tokenizer("Your prompt here", return_tensors="pt").to(device)
    outputs = quantized_model.generate(inputs['input_ids'], max_length=50)
    print(tokenizer.decode(outputs[0], skip_special_tokens=True))

except Exception as e:
    print(f"Error: {e}")
