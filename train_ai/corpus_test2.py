import os
from datasets import Dataset
from transformers import AutoTokenizer
import numpy as np

#CORPUS_DIR = "../corpus/wtk_archive_orig"
CORPUS_DIR = "../corpus/farsight-full"
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

def stream_pack_docs(texts, tok, block_size=BLOCK_SIZE, keep_tail_min=64):
    """
    texts: iterable/list of strings (each is one document/article).
    Returns a Dataset with pretokenized blocks (input_ids, attention_mask).
    """
    buf = []
    blocks = []
    eos_id = tok.eos_token_id

    for doc in texts:
        # tokenize ONE doc at a time (no max_length, no truncation)
        ids = tok(doc, add_special_tokens=False).input_ids
        if eos_id is not None:
            ids = ids + [eos_id]

        # append & flush fixed-size blocks
        buf.extend(ids)
        while len(buf) >= block_size:
            blocks.append({
                "input_ids": buf[:block_size],
                "attention_mask": [1]*block_size,
            })
            buf = buf[block_size:]

    # keep a small tail so we don't throw away data (optional)
    if len(buf) >= keep_tail_min:
        blocks.append({
            "input_ids": buf,
            "attention_mask": [1]*len(buf),
        })

    return Dataset.from_list(blocks)

# Usage:
train_dataset = stream_pack_docs(raw_texts, tok, block_size=BLOCK_SIZE)
print("num blocks:", len(train_dataset))
print("max len:", max(len(x["input_ids"]) for x in train_dataset))  # <= BLOCK_SIZE
print(train_dataset.features)  # expect only input_ids, attention_mask

print(f"Prepared {len(train_dataset)} packed training chunks of {BLOCK_SIZE} tokens ✅")
