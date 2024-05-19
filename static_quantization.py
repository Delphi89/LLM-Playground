import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from torch.quantization import quantize_dynamic, QuantStub, DeQuantStub, get_default_qconfig

# Load your model
model_name = 'meta-llama/Meta-Llama-3-8B'
token = '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'
model = AutoModelForCausalLM.from_pretrained(model_name, token=token)

# Add quantization and dequantization stubs to the model
model.quant = QuantStub()
model.dequant = DeQuantStub()

# Customize quantization configuration for the embedding module
config = get_default_qconfig('fbgemm')
config_dict = {'': config}
config_dict['torch.nn.Embedding'] = torch.quantization.float_qparams_weight_only_qconfig

# Perform static quantization with custom configurations
quantized_model = torch.quantization.quantize(model, config_dict=config_dict, dtype=torch.qint8)

# Save the quantized model
torch.save(quantized_model.state_dict(), './quantized_model.pth')
