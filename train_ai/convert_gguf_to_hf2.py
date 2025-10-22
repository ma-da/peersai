from transformers import AutoTokenizer, AutoModelForCausalLM

# Replace with your GGUF model details
local_model_dir="/storage/models/CWC-Mistral-Nemo/"
gguf_filename = "CWC-Mistral-Nemo-12B-v2-GGUF-q4_k_m.gguf"

# Load tokenizer and model from GGUF
tokenizer = AutoTokenizer.from_pretrained(local_model_dir, gguf_file=gguf_filename)

print(f"Loading model to convert {gguf_filename}")
model = AutoModelForCausalLM.from_pretrained(
    local_model_dir,
    gguf_file=gguf_filename,
)

# Save to a local directory in HF/PyTorch format
output_dir = "converted_model"
tokenizer.save_pretrained(output_dir)
model.save_pretrained(output_dir)