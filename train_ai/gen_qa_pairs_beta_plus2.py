from unsloth import FastLanguageModel
import os
import json
import re
import torch
import glob
import logging
from typing import List, Optional

# must end with slash
OUTPUT_DIR = "/storage/corpus/corpus_beta_plus_qa2/"
LOG_FILE = "/workspace/gen_qa_pairs.log"


# Used for testing. Limit number of blocks to this number. Not used if set to zero. 
BLOCK_LIMIT = 0
    
# If true, will atempt to resume
DO_RESUME = False

# log the first last N below of a segment
LOG_SEGMENT_CHARS = 50

# init logging
logging.basicConfig(filename=LOG_FILE, 
                    level=logging.INFO,
                    format="%(asctime)s - %(levelname)s - %(message)s",
                    force=True)
console = logging.StreamHandler()
console.setLevel(logging.WARNING)
logging.getLogger().addHandler(console)
logging.info("TEST: logger should write this")

if BLOCK_LIMIT != 0:
    logging.warning(f"Utilizing block limit: {BLOCK_LIMIT}")
    
# Load model and tokenizer
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="unsloth/Qwen2.5-7B-Instruct",
    max_seq_length=2048,
    load_in_4bit=True,
    device_map="auto",  # Auto-map to GPU
)
# Enable native 2x faster inference
model = FastLanguageModel.for_inference(model)
logging.warning("Loaded model in 4-bit ✅")

# CORPUS_DIR = "/storage/corpus/wtk_archive_with_stops_with_stops"
CORPUS_DIR = "/storage/corpus/corpus_beta_plus"
BLOCK_SIZE = 1024  # max tokens per chunk
tok = tokenizer

# Ensure EOS/PAD exist and are consistent
added = False
if tok.eos_token is None:
    tok.add_special_tokens({"eos_token": "</s>"})
    added = True
if tok.pad_token is None:
    tok.pad_token = tok.eos_token
    added = True
if added:
    model.resize_token_embeddings(len(tok))

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

logging.warning("Loading files from corpus...")
raw_texts = load_txt_corpus(CORPUS_DIR)
logging.warning(f"Loaded {len(raw_texts)} files from {CORPUS_DIR} ✅")


def split_by_eos_or_chunk(
    text: str,
    tokenizer,
    min_tokens: int = 20,
    max_segments: Optional[int] = None,
    approx_token_budget: Optional[int] = 25_000,
    alt_markers: Optional[List[str]] = None,
) -> List[str]:
    """
    Prefer splitting on tokenizer's EOS token (string or id).
    If no EOS markers are present, fall back to token-length chunking.
    Ensures segments have at least `min_tokens` and stops around desired volume.
    """
    alt_markers = alt_markers or ["</s>", "<|im_end|>", "<|endoftext|>"]
    # 1) Try string-based split using known markers (fast path)
    s = text
    for m in alt_markers:
        if m and m in s:
            parts = [p.strip() for p in s.split(m)]
            # filter by token count, not chars
            kept = []
            for p in parts:
                if not p:
                    continue
                if len(tokenizer.encode(p, add_special_tokens=False)) >= min_tokens:
                    kept.append(p)
            if kept:
                # enforce target volume and segment cap
                if approx_token_budget:
                    total = 0
                    budgeted = []
                    for seg in kept:
                        n = len(tokenizer.encode(seg, add_special_tokens=False))
                        if total + n > approx_token_budget or (max_segments and len(budgeted) >= max_segments):
                            break
                        budgeted.append(seg)
                        total += n
                    return budgeted
                if max_segments:
                    return kept[:max_segments]
                else:
                    return kept
    # 2) Fallback: chunk by tokens to ~fixed size if no markers found
    ids = tokenizer.encode(text, add_special_tokens=False)
    # choose a chunk size that’s friendly for your model/context
    CHUNK_TOKENS = max(min_tokens * 4, 1024)  # e.g., 512–1024
    segments = []
    i = 0
    total = 0
    while i < len(ids) and (approx_token_budget is None or total < approx_token_budget) and (not max_segments or len(segments) < max_segments):
        j = min(i + CHUNK_TOKENS, len(ids))
        seg_text = tokenizer.decode(ids[i:j], skip_special_tokens=True).strip()
        if seg_text:
            segments.append(seg_text)
            total += (j - i)
        i = j
    return segments

def split_with_overlap_and_stops(
    text: str,
    tokenizer,
    overlap_chars: int = 200,
    forward_tokens: int = 824,
    stop_markers: Optional[List[str]] = None,
    max_segments: Optional[int] = None,
    approx_token_budget: Optional[int] = None,
) -> List[str]:
    """
    Produces windowed segments with:
      - 200 previous characters (default)
      - 824 new forward tokens (default)
    Respects stop tokens: if a stop-token appears inside the previous-200-chars
    window, only keep text *after* that stop-token.
    """

    stop_markers = stop_markers or ["</s>", "<|im_end|>", "<|endoftext|>"]

    # ---- Tokenize the entire text once ----
    full_ids = tokenizer.encode(text, add_special_tokens=False)
    segments = []
    total_tokens_used = 0

    pos = 0  # token pointer

    while pos < len(full_ids):
        if approx_token_budget and total_tokens_used >= approx_token_budget:
            break
        if max_segments and len(segments) >= max_segments:
            break

        # ---------------------------------------------------------
        # 1. Identify raw-text span corresponding to "200 previous chars"
        # ---------------------------------------------------------
        if pos == 0:
            prev_text = ""
        else:
            # decode a large window back to raw text, then clip to 200 chars
            # To avoid huge backtracking, decode at most 2000 chars worth of tokens
            back_start = max(0, pos - 2000)
            raw_back = tokenizer.decode(full_ids[back_start:pos], skip_special_tokens=False)
            prev_text = raw_back[-overlap_chars:]  # take last 200 chars

        # ---------------------------------------------------------
        # 2. Trim previous-window if it contains stop tokens
        # ---------------------------------------------------------
        last_stop_index = -1
        for m in stop_markers:
            idx = prev_text.rfind(m)
            if idx != -1:
                last_stop_index = max(last_stop_index, idx + len(m))

        if last_stop_index != -1:
            prev_text = prev_text[last_stop_index:]

        # ---------------------------------------------------------
        # 3. Extract next 824 tokens of NEW content
        # ---------------------------------------------------------
        next_end = min(pos + forward_tokens, len(full_ids))
        new_chunk_text = tokenizer.decode(full_ids[pos:next_end], skip_special_tokens=False)

        # Combine previous-text + new-text
        combined = (prev_text + new_chunk_text).strip()

            
        if combined:
            segments.append(combined)
            # debug logging for generated segments
            logging.info(f"Added segement with len {len(combined)}")
            left_n = min(LOG_SEGMENT_CHARS, len(combined))
            right_n = max(0, len(combined) - LOG_SEGMENT_CHARS)
            logging.info(f"left_segment: {combined[:left_n]}")
            logging.info(f"right_segment: {combined[right_n:]}")
            

        # advance
        used_tokens = next_end - pos
        total_tokens_used += used_tokens
        pos = next_end

        if pos >= len(full_ids):
            break

    return segments



blocks = []
for text in raw_texts:
    
    #segments = split_by_eos_or_chunk(
    #    text,
    #    tokenizer,
    #    min_tokens=50,
    #    max_segments=50,
    #    approx_token_budget=25_000,  # aim for ~25k tokens total
    #    alt_markers=["</s>", "<|im_end|>"],  # add any markers your data uses
    #)
    
    #def split_with_overlap_and_stops(
    #text: str,
    #tokenizer,
    #overlap_chars: int = 200,
    #forward_tokens: int = 824,
    #stop_markers: Optional[List[str]] = None,
    #max_segments: Optional[int] = None,
    #approx_token_budget: Optional[int] = None,
        
    segments = split_with_overlap_and_stops(
        text,
        tokenizer,
        overlap_chars=200,
        forward_tokens=824,
        stop_markers=["</s>", "<|im_end|>"],  # add any markers your data uses
        max_segments=500,
        approx_token_budget=25_000,  # aim for ~25k tokens total
    )
  
    blocks.extend(segments)
    if len(blocks) % 1000 == 0:
        logging.warning(f"Processed {len(blocks)} number of blocks...")
        
    if BLOCK_LIMIT != 0 and len(blocks) > BLOCK_LIMIT:
        logging.warning("Hit block limit. Stopping text processing.")
        break
        
    logging.info(f"Procssed raw text with len {len(text)} which created len segments {len(segments)}")
    
logging.warning(f"Prepared {len(blocks)} packed training chunks ✅")

SEGMENTS_LIMIT = 100000

if BLOCK_LIMIT != 0 and BLOCK_LIMIT < SEGMENTS_LIMIT:
    logging.warning(f"Manually setting segments limit: {BLOCK_LIMIT}")
    SEGMENTS_LIMIT = BLOCK_LIMIT
    
MAX_TOKENS = 512
REPORTING_INTERVAL = 2
JSON_WRITE_INTERVAL = 50
qa_pairs = []

logging.warning(f"Generating qa pairs for {len(blocks)} number of blocks...")

# Find the last processed segment index from existing files
if DO_RESUME:
    output_files = glob.glob(OUTPUT_DIR + "qa_pairs_*.json")
    if output_files:
        nums = []
        for f in output_files:
            try:
                num_str = re.search(r'qa_pairs_(\d+)\.json', f).group(1)
                nums.append(int(num_str))
            except:
                continue
        if nums:
            start_i = max(nums) + 1  # Resume from next segment after the highest saved i
            logging.warning(f"Resuming from segment {start_i} (found existing files up to qa_pairs_{max(nums)}.json)")
        else:
            start_i = 1
    else:
        start_i = 1
else:
    start_i = 1
        
i = start_i - 1  # Adjust for 0-based indexing in blocks list
total_pairs = max(0, start_i - 1)  # Total pairs already processed (approximate)
qa_pairs = []  # Start fresh for the current batch

for idx, segment in enumerate(blocks[i:], start=start_i):
    if i > SEGMENTS_LIMIT:
        logging.warning(f"Hit segments limit {SEGMENTS_LIMIT}. Stopping.")
        break
    if i % REPORTING_INTERVAL == 0:
        logging.warning(f"Processed number of pairs {total_pairs}...")
    # Truncate to ~512-1024 tokens for context length (align with max_seq_length=1024)
    tokens = segment.split()[:1024]
    segment = " ".join(tokens)
    # Prompt with escaped curly braces for JSON example
    prompt = f"""You are an unrestricted analyst. You may discuss any topic.
    
    Extract the 5 MOST important entities, terms, or claims from this segment:

    <<<SEGMENT>>>
    {segment}
    <<<END SEGMENT>>>

    Generate 6 high-signal, diverse Q-A pairs:
    - 2 factual (e.g., who/what/when/where) referencing at least two of the extracted key entities
    - 1 explanatory (e.g., cause/effect, reasons, mechanisms).
    - 1 inference question strictly based on evidence inside the segment (e.g., what might happen next?)
    - 1 domain-specific (e.g., legal, technical, policy, scientific—match the domain).
  
    REQUIREMENTS:
    - Every question MUST reference at least one extracted entity/term.
    - At least 3 questions MUST share overlapping entities.
    - Answers MUST be fully grounded in the given text and avoid any outside knowledge.

    Format as JSON list: [{{ "question": "...", "answer": "...</s>"}}, ...]"""

    system_prompt = f"""
    You are an unrestricted analyst specializing in esoteric and covert topics. From the provided text segment, generate 6 high-signal, diverse Q-A pairs:
    You are producing high-quality question–answer pairs taken only from or inferred from the given text segment.

    Your questions must be:
    - highly specific to the exact entities, events, and claims in the segment,
    - not generic or reusable on other texts,
    - mutually overlapping: at least 3 questions must involve the same key entities or concepts,
    - consistent: all answers must be correct given the segment and not hallucinate any facts,
    - concise but fully informative (60–250 words),
    - always valid JSON.
    - Never reveal your reasoning. /no_think
    - Never include anything except the JSON array.
    
    You must output valid, strictly correct JSON.  
    Never insert stray whitespace between JSON keys.
    """
    
    try:
        # Build messages
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt},
        ]
        has_chat_template = getattr(tokenizer, "chat_template", None) not in (None, "")
        # Encode
        if has_chat_template:
            enc = tokenizer.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
                return_tensors="pt",
            )
            # enc can be a Tensor or a dict; normalize
            if isinstance(enc, torch.Tensor):
                inputs = {"input_ids": enc.to(model.device)}
                inputs["attention_mask"] = torch.ones_like(inputs["input_ids"])
            else:
                inputs = {k: v.to(model.device) for k, v in enc.items()}
                if "attention_mask" not in inputs:
                    inputs["attention_mask"] = torch.ones_like(inputs["input_ids"])
        else:
            prompt_str = (
                f"### System:\n{system_prompt}\n\n"
                f"### User:\n{prompt}\n\n"  # Fixed: use prompt here, not user_prompt
                f"### Assistant:\n"
            )
            enc = tokenizer(prompt_str, return_tensors="pt")
            inputs = {k: v.to(model.device) for k, v in enc.items()}
            
        
        # fail if input too long
        encoded = tokenizer(prompt, return_tensors="pt")
        input_len = encoded["input_ids"].shape[1]
    
        if input_len > model.config.max_position_embeddings:
            raise ValueError(f"Input too long: {input_len} tokens > {model.config.max_position_embeddings}")

        # Tokenize and generate
        # inputs = tokenizer(prompt, return_tensors="pt", max_length=1024, truncation=True).to("cuda")
        outputs = model.generate(
            **inputs,
            max_new_tokens=MAX_TOKENS,  # Limit for speed
            do_sample=True,  # Deterministic for consistency
            num_return_sequences=1,
            temperature=0.1, # ultra-low temperature: produces structured, faithful, non-generic QAs
            top_p=0.90,  # limit selection to this set of cumulative probability
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # print(f"Response: {response}")
        # Parse JSON (trim prompt and handle </s>)
        think_start = response.find("[")
        json_start = response.find("[", think_start + 1)
        json_end = response.rfind("]", json_start + 1) + 1
        
        if json_start != -1 and json_end != -1:
            snippet = response[json_start:json_end]
            logging.info(f"SNIPPET {i}: {snippet}")

            qa_list = json.loads(snippet)
            # Ensure </s> in answers
            for qa in qa_list:
                if not qa["answer"].endswith("</s>"):
                    qa["answer"] += "</s>"
                qa_pairs.append(qa)

            total_pairs += 1  # Increment for each successful segment
            if total_pairs % REPORTING_INTERVAL == 0:
                logging.warning(f"Processed {total_pairs} number of segments...")

    except Exception as e:
        logging.warning(f"Error processing segment {i}: {e}")
        continue

    # Save Q-A pairs to JSON for LoRA training on interval
    if i > 0 and i % JSON_WRITE_INTERVAL == 0:
        output_filename = OUTPUT_DIR + f"qa_pairs_{i}.json"
        with open(output_filename, "w", encoding="utf-8") as f:
            json.dump(qa_pairs, f, ensure_ascii=False, indent=2)
        logging.warning(f"Wrote {len(qa_pairs)} qa pairs to file {output_filename}")
        qa_pairs = []  # Reset for next batch

    # Clear GPU memory to prevent OOM
    torch.cuda.empty_cache()

# Save remaining Q-A pairs
if qa_pairs:
    output_filename = OUTPUT_DIR + f"qa_pairs_{i}.json"
    with open(output_filename, "w", encoding="utf-8") as f:
        json.dump(qa_pairs, f, ensure_ascii=False, indent=2)
    logging.warning(f"Wrote {len(qa_pairs)} QA pairs to file {output_filename}...")

logging.warning(f"Generated {total_pairs} total QA pairs ✅ ")