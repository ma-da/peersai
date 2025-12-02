from unsloth import FastLanguageModel
import os
import json
import re
import torch
import glob
import time
import logging
from typing import List, Optional
from jsonschema import validate, ValidationError


# =============================================================================
# CONFIGURATION
# =============================================================================
OUTPUT_DIR = "/storage/corpus/corpus_beta_plus_qa2/"  # Must end with slash
os.makedirs(OUTPUT_DIR, exist_ok=True)

CORPUS_DIR = "/storage/corpus/corpus_beta_plus"

LOG_FILE = "/workspace/gen_qa_pairs.log"


USE_RESUME = False

# Used for testing. Limit number of blocks to this number. Not used if set to zero. 
BLOCK_LIMIT = 3

# Sliding window settings — best balance of quality + quantity + low duplication
CHUNK_TOKENS   = 896    # ~90% of context, leaves room for prompt
OVERLAP_TOKENS = 256    # ~30% overlap → smooth continuity, minimal duplication

MAX_TOKENS_GENERATION = 1024
JSON_WRITE_INTERVAL   = 1
REPORTING_INTERVAL    = 10
SEGMENTS_LIMIT        = 500_000

# init logging
logging.basicConfig(filename=LOG_FILE, 
                    level=logging.INFO,
                    format="%(asctime)s - %(levelname)s - %(message)s",
                    filemode='w',
                    force=True)
console = logging.StreamHandler()
console.setLevel(logging.WARNING)
logging.getLogger().addHandler(console)
logging.info("TEST: logger should write this")

if BLOCK_LIMIT != 0:
    logging.warning(f"Utilizing block limit: {BLOCK_LIMIT}")
    

# =============================================================================
# HELPERS
# =============================================================================

response_schema = {
    "type": "array",
    "items": {
        "type": "object",
        "properties": {
            "question": {"type": "string"},
            "answer": {"type": "string"},
        },
        "required": ["question", "answer"]
    }
}

def is_valid_json_with_schema(text):
    try:
        data = json.loads(text)
        validate(instance=data, schema=response_schema)
        return True
    except (ValueError, ValidationError):
        return False
    

# =============================================================================
# MODEL LOADING
# =============================================================================
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="unsloth/Qwen2.5-7B-Instruct",
    max_seq_length=2048,
    load_in_4bit=True,
    device_map="auto",
)
model = FastLanguageModel.for_inference(model)
print("Loaded model in 4-bit")

# Ensure EOS/PAD
if tokenizer.eos_token is None:
    tokenizer.add_special_tokens({"eos_token": "</s>"})
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
if tokenizer.eos_token_id is None or tokenizer.pad_token_id is None:
    model.resize_token_embeddings(len(tokenizer))

# =============================================================================
# LOAD CORPUS (deterministic order!)
# =============================================================================
def load_txt_corpus(directory: str) -> List[str]:
    txt_files = [f for f in os.listdir(directory) if f.lower().endswith(".txt")]
    txt_files.sort()  # ← CRITICAL: deterministic order every run
    texts = []
    for filename in txt_files:
        path = os.path.join(directory, filename)
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            content = f.read().strip()
            if content:
                texts.append(content)
        #print(f"Loaded {filename}")
    return texts

raw_texts = load_txt_corpus(CORPUS_DIR)
print(f"Loaded {len(raw_texts)} files from {CORPUS_DIR}")

# =============================================================================
# SLIDING WINDOW CHUNKER (with overlap)
# =============================================================================
def sliding_window_chunks(
    text: str,
    tokenizer,
    chunk_tokens: int = CHUNK_TOKENS,
    overlap_tokens: int = OVERLAP_TOKENS,
    min_tokens: int = 50,
) -> List[str]:
    ids = tokenizer.encode(text, add_special_tokens=False)
    if len(ids) <= chunk_tokens:
        seg = tokenizer.decode(ids, skip_special_tokens=True).strip()
        return [seg] if len(tokenizer.encode(seg, add_special_tokens=False)) >= min_tokens else []

    chunks = []
    i = 0
    while i < len(ids):
        j = min(i + chunk_tokens, len(ids))
        seg_text = tokenizer.decode(ids[i:j], skip_special_tokens=True).strip()
        if len(tokenizer.encode(seg_text, add_special_tokens=False)) >= min_tokens:
            chunks.append(seg_text)
        if j == len(ids):
            break
        i = j - overlap_tokens  # ← overlap
    return chunks

# Build blocks with sliding windows
blocks = []
for text in raw_texts:
    segments = sliding_window_chunks(text, tokenizer)
    blocks.extend(segments)
    if len(blocks) % 1000 == 0:
        print(f"→ Created {len(blocks)} segments so far...")
    if BLOCK_LIMIT != 0 and len(blocks) >= BLOCK_LIMIT:
        print(f"Hit block limit of num blocks {BLOCK_LIMIT}. Exiting block processing...")
        break

print(f"Prepared {len(blocks)} overlapping segments (CHUNK={CHUNK_TOKENS}, OVERLAP={OVERLAP_TOKENS})")

# =============================================================================
# RESUME LOGIC — find last processed segment
# =============================================================================
total_qa_pairs_generated = 0
successful_segments = 0
start_i = 1

if USE_RESUME:
    output_files = glob.glob(OUTPUT_DIR + "qa_pairs_*.json")
    if output_files:
        nums = [int(re.search(r'qa_pairs_(\d+)\.json', f).group(1)) for f in output_files if re.search(r'qa_pairs_(\d+)\.json', f)]
        start_i = max(nums) + 1 if nums else 1
        for f in output_files:
            try:
                with open(f, 'r') as jf:
                    data = json.load(jf)
                    total_qa_pairs_generated += len(data)
                    successful_segments += 1  # Assume 1 file per batch of successes
            except:
                print(f"Warning: Corrupt file {f}")
        logging.warning(f"Resuming from segment {start_i} with {total_qa_pairs_generated} prior pairs")
    else:
        logging.warning("Resuming is disabled.")
    
i = start_i - 1
qa_pairs = []

# Stopping criteria 
from transformers import StoppingCriteria, StoppingCriteriaList
class StopOnStrings(StoppingCriteria):
    def __init__(self, tokenizer, stop_strings: List[str]):
        self.stop_ids = [tokenizer.encode(s, add_special_tokens=False) for s in stop_strings if s]
    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        seq = input_ids[0].tolist()
        for ids in self.stop_ids:
            if len(seq) >= len(ids) and seq[-len(ids):] == ids:
                return True
        return False

# =============================================================================
# PROMPT TEMPLATES (nuclear-grade truth extraction)
# =============================================================================


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

# =============================================================================
# MAIN GENERATION LOOP
# =============================================================================
logging.warning(f"Generating QA pairs starting from segment {start_i}...")
num_errors = 0
for idx, segment in enumerate(blocks[i:], start=start_i):
    i = idx
    if i > SEGMENTS_LIMIT:
        logging.warning(f"Hit segments limit {SEGMENTS_LIMIT}. Stopping.")
        break
    if i % REPORTING_INTERVAL == 0:
        logging.warning(f"Processing segment {i}...")
        
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
    - Questions and answers must NOT use brackets or braces characters inside their text.
    - Output MUST be a single JSON list: [{{"question":"...","answer":"..."}}]
    - No preamble, no commentary, no trailing text.
    - Do not use newlines within question or answer texts.
    - Output ONLY valid JSON.
    
    Example format (follow exactly):
    [
      {{"question": "Who founded the Autistic Self-Advocacy Network?", "answer": "Ari Ne'eman founded ASAN in 2006."}},
      {{"question": "What does 'Nothing About Us Without Us' demand?", "answer": "It demands that autistic people must be included in all decisions about autism."}},
      {{"question": "How did Ari Ne'eman influence federal autism policy?", "answer": "He was the first autistic person appointed to the IACC federal committee."}},
      {{"question": "Why do neurodiversity advocates oppose cure-focused research?", "answer": "They view autism as a neurological difference, not a disease to be cured."}},
      {{"question": "What risk does the neurodiversity movement pose to traditional autism organizations?", "answer": "It threatens their funding and influence by rejecting the medical model of autism."}},
      {{"question": "If neurodiversity policies are fully adopted, what might happen to ABA therapy?", "answer": "ABA would likely be restricted or banned as a human rights violation."}}
    ]
    """

    # Dynamic trim
    prompt_overhead = len(tokenizer.encode(prompt)) + 100  # Buffer
    max_seg_tokens = 1024 - prompt_overhead
    seg_tokens = tokenizer.encode(segment, add_special_tokens=False)
    if len(seg_tokens) > max_seg_tokens:
        segment = tokenizer.decode(seg_tokens[:max_seg_tokens], skip_special_tokens=True)

    response = ""
    try:
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt},
        ]
        enc = tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
        ).to(model.device)
        inputs = {"input_ids": enc} if isinstance(enc, torch.Tensor) else enc
        if "attention_mask" not in inputs:
            inputs["attention_mask"] = torch.ones_like(inputs["input_ids"])
            

        #stopper = StoppingCriteriaList([StopOnStrings(tokenizer, ["</s>", "<|im_end|>", "<|endoftext|>", "\n\n"])])
        
        stopper = StoppingCriteriaList([
            StopOnStrings(tokenizer, [
                "]",           # ← PRIMARY — stops the instant the array closes
                "]</s>",       # ← backup if it adds </s> after ]
                "\n]",         # ← catches line-break versions
                "},\n]",       # ← catches formatted versions
                "</s>",        # ← generic EOS fallback
                "<|im_end|>", 
                "<|endoftext|>"
            ])
        ])
        
        # This should fix random emojis and provide deterministic output.
        outputs = model.generate(
            **inputs,
            max_new_tokens=1024,
            do_sample=False,           # ← THE FIX
            temperature=0.0,           # ← ignored when do_sample=False
            repetition_penalty=1.2,    # ← still useful
            num_beams=1,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            stopping_criteria=stopper,
        )
        
        # These are previous settings I used. But sometimes random emojis would appear.
        #outputs = model.generate(
        #    **inputs,
        #    max_new_tokens=MAX_TOKENS_GENERATION,
        #    do_sample=True,  # Better for JSON
        #    temperature=0.3,
        #    top_p=0.95,
        #    num_beams=1,  # Beam for structured output
        #    repetition_penalty=1.18,  # helps with bad json outputs
        #    pad_token_id=tokenizer.pad_token_id,
        #    eos_token_id=tokenizer.eos_token_id,
        #    stopping_criteria=stopper,
        #)
                
        #response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # This is the gold standard — decode ONLY the new tokens
        #generated_tokens = outputs[0][inputs["input_ids"].shape[-1]:]  # skip prompt
        #clean_response = tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()
        
        # Find where the assistant's response actually starts
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)

        logging.info(f"*** Begin Raw response ***\n {response}")
        logging.info(f"*** End Raw response ***")
        
        # Everything before the final "assistant" (or equivalent) is the prompt
        assistant_pos = response.rfind("assistant")  # Qwen2.5 uses plain "assistant"
        if assistant_pos != -1:
            clean_response = response[assistant_pos + len("assistant"):].strip()
        else:
            # Fallback: try other common markers
            for marker in ["\nassistant\n", "<|assistant|>", "### Assistant:"]:
                pos = response.rfind(marker)
                if pos != -1:
                    clean_response = response[pos + len(marker):].strip()
                    break
            else:
                clean_response = response  # give up, return all
        
        response = clean_response
        logging.info(f"*** Begin Cleaned response ***\n {response}")
        logging.info(f"*** End Cleaned response ***")
        
        # Validty check
        #if is_valid_json_with_schema(response):
        #    logging.info("Response was valid json")
        #else:
        #    logging.error("Response was NOT valid json")
           
        # Balanced extraction
        start_idx = response.find('[')
        if start_idx == -1:
            logging.error(f"Invalid json. No [ found.")
            raise ValueError("No [ found")
                            
        count = 0
        end_idx = -1
        for pos in range(start_idx, len(response)):
            if response[pos] == '[':
                count += 1
            elif response[pos] == ']':
                count -= 1
                if count == 0:
                    end_idx = pos
                    break
        if end_idx == -1:
            logging.error(f"Invalid json. Unbalanced brackets.")
            raise ValueError("Unbalanced brackets")
        logging.info(f"Snippet start {start_idx} and end {end_idx}")
        snippet = response[start_idx:end_idx+1].strip()
        
        logging.info(f"Snippet: {snippet}")
        qa_list = json.loads(snippet)
        
        qa_num = 1
        for qa in qa_list:
            q = qa["question"].strip()
            ans = qa["answer"].strip()
            logging.info(f"QA {qa_num} question: {q}")
            logging.info(f"QA {qa_num} ans: {ans}")
            if not ans.endswith("</s>"):
                qa["answer"] = ans + "</s>"  # No extra space
            qa_pairs.append(qa)
            qa_num = qa_num + 1
            
        successful_segments += 1
        total_qa_pairs_generated += len(qa_list)
        logging.info(f"Segment {i}: Generated {len(qa_list)} QA pairs (total: {total_qa_pairs_generated})")
    except Exception as e:
        logging.error(f"Error on segment {i}: {e}")
        num_errors = num_errors + 1
        with open(f"{OUTPUT_DIR}failed_segment_{i}.txt", "w") as f:
            f.write(response)
        logging.info(f"Saved failed response to failed_segment_{i}.txt")
        # Partial save on error
        if qa_pairs:
            err_filename = f"{OUTPUT_DIR}qa_pairs_partial_{i}.json"
            with open(err_filename, "w") as f:
                json.dump(qa_pairs, f, ensure_ascii=False, indent=2)
            logging.warning(f"Partial save on error: {len(qa_pairs)} pairs → {err_filename}")
            qa_pairs = []
        continue
        
    # Periodic save (unchanged)
    if i > 0 and i % JSON_WRITE_INTERVAL == 0:
        filename = f"{OUTPUT_DIR}qa_pairs_{i}.json"
        with open(filename, "w", encoding="utf-8") as f:
            json.dump(qa_pairs, f, ensure_ascii=False, indent=2)
        logging.warning(f"Saved {len(qa_pairs)} pairs → {filename}")
        qa_pairs = []
    torch.cuda.empty_cache()  # No sleep

# Final save (with failure rate)
if qa_pairs:
    filename = f"{OUTPUT_DIR}qa_pairs_{i}.json"
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(qa_pairs, f, ensure_ascii=False, indent=2)
    logging.warning(f"Final save: {len(qa_pairs)} pairs → {filename}")

total_segments_processed = i - start_i + 1
failure_rate = 1 - (successful_segments / total_segments_processed) if total_segments_processed > 0 else 0

print(f"Done! Generated {total_qa_pairs_generated} QA pairs from {successful_segments} successful segments (failure rate: {failure_rate:.2%}, num_errors: {num_errors}).")