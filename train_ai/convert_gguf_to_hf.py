import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from huggingface_hub import HfApi


# Load
model_id = "TheBloke/Llama-2-7B-Chat-GGUF"
filename = "llama-2-7b-chat.Q5_K_M.gguf"
tokenizer = AutoTokenizer.from_pretrained(model_id, gguf_file=filename)
model = AutoModelForCausalLM.from_pretrained(model_id, gguf_file=filename, torch_dtype=torch.float16)

# Optional: Dequantize to full precision
# model = model.to(dtype=torch.float32)

# Save locally
output_dir = "./llama2-hf-converted"
model.save_pretrained(output_dir, safe_serialization=True)
tokenizer.save_pretrained(output_dir)

# Optional: Upload
#api = HfApi()
#api.upload_folder(folder_path=output_dir, repo_id="your-username/llama2-hf-converted", repo_type="model")
#print("Conversion complete! Model saved to", output_dir)