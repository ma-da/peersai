import os
from datasets import Dataset
from transformers import AutoTokenizer
import numpy as np

CORPUS_DIR = "../corpus/wtk_archive_orig"
#CORPUS_DIR = "../corpus/farsight-full"
BLOCK_SIZE = 1024  # max tokens per chunk

tok = AutoTokenizer.from_pretrained("sshleifer/tiny-gpt2")

# -----------------------
# 2) Load raw text files (no EOS strings here)
# -----------------------
def load_txt_corpus(directory):
    texts = []
    for filename in os.listdir(directory):
        if not filename.endswith(".txt"):
            continue
        path = os.path.join(directory, filename)
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            txt = f.read().strip()
            if txt:
                texts.append(txt)
    return texts

raw_texts = load_txt_corpus(CORPUS_DIR)
print(f"Loaded {len(raw_texts)} files from {CORPUS_DIR} ✅")

assert isinstance(raw_texts, (list, tuple)) and isinstance(raw_texts[0], str)

total_size = 0
for text in raw_texts:
    total_size = total_size + len(text)
print(f"Raw size length {total_size}")

def tokenize_append_eos(texts):
    # batch tokenize; append eos token string so tokenizer emits eos_token_id
    # Alternatively: add eos id manually after each example (equivalent).
    enc = tok(texts, add_special_tokens=False)
    input_ids = []
    for ids in enc["input_ids"]:
        input_ids.extend(ids)
        if tok.eos_token_id is not None:
            input_ids.append(tok.eos_token_id)
    return input_ids

flat_ids = tokenize_append_eos(raw_texts)

arr = np.array(flat_ids, dtype=np.int32)
print("flat_ids logical size in bytes:", arr.nbytes)

# -----------------------
# 4) Pack into fixed-length blocks (no cross-doc bleed because we injected EOS)
# -----------------------
#def pack_ids_to_blocks(ids, block_size):
#    blocks = []
#    for i in range(0, len(ids) - block_size + 1, block_size):
#        chunk = ids[i : i + block_size]
#        blocks.append({"input_ids": chunk, "attention_mask": [1] * len(chunk)})
#    return Dataset.from_list(blocks)

def pack_ids_to_blocks(ids, block_size, keep_tail_min=64):
    blocks = []
    for i in range(0, len(ids) - block_size + 1, block_size):
        chunk = ids[i:i+block_size]
        blocks.append({"input_ids": chunk, "attention_mask": [1]*block_size})
    tail = len(ids) % block_size
    if tail >= keep_tail_min:
        t = ids[-tail:]
        blocks.append({"input_ids": t, "attention_mask": [1]*tail})
    return Dataset.from_list(blocks)

train_dataset = pack_ids_to_blocks(flat_ids, BLOCK_SIZE)


total_size = 0
for block in train_dataset:
    total_size = total_size + len(block)
print(f"Train dataset length {total_size}")

print(f"Prepared {len(train_dataset)} packed training chunks of {BLOCK_SIZE} tokens ✅")
