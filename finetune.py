import requests
from bs4 import BeautifulSoup
from datasets import load_dataset
from transformers import AutoTokenizer, LlamaForCausalLM, Trainer, TrainingArguments
from huggingface_hub import login
import torch

torch.cuda.empty_cache()

login('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')


# Scrape data from the website
url = 'https://cnimslatina.ro/despre'
response = requests.get(url)
soup = BeautifulSoup(response.text, 'html.parser')

# Extract text and save to a file
data = soup.get_text()
with open('data.txt', 'w') as f:
    f.write(data)

# Prepare the dataset
dataset = load_dataset('text', data_files={'train': 'data.txt'})

# Load the pre-trained LLaMA model and tokenizer with authentication
model_name = 'meta-llama/Meta-Llama-3-8B'
token = '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'

# Load tokenizer and model with the token
tokenizer = AutoTokenizer.from_pretrained(model_name, token=token)
model = LlamaForCausalLM.from_pretrained(model_name, token=token)

# Tokenize the dataset
def tokenize_function(examples):
    return tokenizer(examples['text'], truncation=True, padding='max_length')

tokenized_datasets = dataset.map(tokenize_function, batched=True)

# Set up training arguments and trainer with reduced batch size and mixed precision
training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=3,
    per_device_train_batch_size=1,  # Reduce batch size
    save_steps=10_000,
    save_total_limit=2,
    fp16=True,  # Enable mixed precision training
    gradient_accumulation_steps=16,  # Accumulate gradients
)

# Check if CUDA is available and set the device accordingly
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('\n Using device:', device)
#Additional Info when using cuda
if device.type == 'cuda':
    print(torch.cuda.get_device_name(0))
    print('Memory Usage:')
    print('Allocated:', round(torch.cuda.memory_allocated(0)/1024**3,1), 'GB')
    print('Cached:   ', round(torch.cuda.memory_reserved(0)/1024**3,1), 'GB')
print()
model = model.to(device)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets['train'],
)

print("pretrain 2\n")

# Train the model
trainer.train()

# Save the fine-tuned model
model.save_pretrained('./fine-tuned-llama')
tokenizer.save_pretrained('./fine-tuned-llama')

# Evaluate the model
inputs = tokenizer("Your prompt here", return_tensors="pt")
outputs = model.generate(inputs['input_ids'], max_length=50)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
