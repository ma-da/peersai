# Install dependencies first:
# pip install unsloth datasets accelerate bitsandbytes

import os
import torch
from datasets import load_dataset
from unsloth import FastLanguageModel

# -------------------------------
# 1. Select Device
# -------------------------------
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"Using device: {device}")

# -------------------------------
# 2. Model & Checkpoint Settings
# -------------------------------
model_name = "unsloth/llama-2-7b"  # Replace with your model
checkpoint_dir = "lora-checkpoint" # Directory to save intermediate checkpoints
resume_from_checkpoint = True      # Set to False to start fresh

# -------------------------------
# 3. Load Model in 4-bit (QLoRA)
# -------------------------------
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = model_name,
    load_in_4bit = True,           # QLoRA memory savings
    device_map = {"": device},     # MPS device mapping
)

# -------------------------------
# 4. Add LoRA Adapters (or Resume)
# -------------------------------
if resume_from_checkpoint and os.path.exists(checkpoint_dir):
    print(f"Resuming from checkpoint: {checkpoint_dir}")
    model.load_adapter(checkpoint_dir)
else:
    print("No checkpoint found. Initializing new LoRA adapters.")
    model = FastLanguageModel.get_peft_model(
        model,
        r = 8,
        lora_alpha = 16,
        target_modules = ["q_proj", "v_proj"],
        lora_dropout = 0.05,
        bias = "none",
        task_type = "CAUSAL_LM"
    )

# -------------------------------
# 5. Load Dataset
# -------------------------------
dataset = load_dataset("tatsu-lab/alpaca")

def format_example(example):
    return {
        "text": f"### Instruction:\n{example['instruction']}\n\n### Response:\n{example['output']}"
    }

dataset = dataset.map(format_example)

# -------------------------------
# 6. Fine-Tune with Checkpoint Saving
# -------------------------------
model.fit(
    dataset["train"],
    tokenizer = tokenizer,
    epochs = 3,
    batch_size = 1,
    max_seq_length = 512,
    learning_rate = 2e-4,
    save_steps = 200,  # Save checkpoint every N steps
    save_dir = checkpoint_dir
)

# -------------------------------
# 7. Save Final LoRA
# -------------------------------
model.save_pretrained("lora-out")
tokenizer.save_pretrained("lora-out")

print("✅ LoRA fine-tuning complete. Final model saved to 'lora-out'")
