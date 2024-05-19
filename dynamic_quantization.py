import torch
import torch.quantization
from transformers import AutoModelForCausalLM, AutoTokenizer
from huggingface_hub import login

# Log in with your Hugging Face token
login('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')

# Load your model
model_name = 'meta-llama/Meta-Llama-3-8B'
token = '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'
model = AutoModelForCausalLM.from_pretrained(model_name, token=token)

# Set the model to evaluation mode
model.eval()

# Quantize the model
quantized_model = torch.quantization.quantize_dynamic(
    model, {torch.nn.Linear}, dtype=torch.qint8
)

# Save the quantized model
torch.save(quantized_model.state_dict(), './quantized_model.pth')

