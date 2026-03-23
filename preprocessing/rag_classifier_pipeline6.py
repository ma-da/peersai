import argparse
import spacy
from sentence_transformers import SentenceTransformer
from rag_topics import *
import re
import json
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import torch
import numpy as np

model_id = "meta-llama/Llama-3.1-8B-Instruct"

tokenizer = AutoTokenizer.from_pretrained(model_id, padding_side="left")
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# 4-bit config
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,   # or bfloat16 if supported
    bnb_4bit_quant_type="nf4"               # best quality
)

chat_model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype="auto",
    quantization_config=bnb_config,
    device_map="auto",
)


model = SentenceTransformer("BAAI/bge-large-en")

domain_vectors = {
    d: model.encode(desc, normalize_embeddings=True)
    for d, desc in DOMAINS.items()
}

def classify_domain_fast(text):
    vec = model.encode(text, normalize_embeddings=True)

    best_domain = None
    best_score = -1

    for d, dv in domain_vectors.items():
        score = (vec @ dv)
        if score > best_score:
            best_score = score
            best_domain = d

    return best_domain


def classify_domain_batch(texts):
    vecs = model.encode(texts, normalize_embeddings=True)

    domains = []
    scores = []

    domain_keys = list(domain_vectors.keys())
    domain_matrix = np.array([domain_vectors[d] for d in domain_keys])

    sims = vecs @ domain_matrix.T  # [batch, domains]

    for row in sims:
        idx = np.argmax(row)
        domains.append(domain_keys[idx])
        scores.append(row[idx])

    return domains, scores


def should_use_llm(score, threshold=0.80):
    return True
    #return score < threshold

#llm = Llama(
#    model_path=MODEL_PATH,
#    n_ctx=4096,
#    n_threads=8,
#    temperature=0.1
#)

PROMPT_TEMPLATE = """
You are a classification system.

Your task is to analyze the text and determine:

1. The most appropriate DOMAIN.
2. Up to 3 TOPICS that best describe the text.

Rules:
- DOMAIN must be chosen ONLY from the provided domain list.
- TOPICS must be chosen ONLY from the provided topic list.
- Return a maximum of 3 topics.
- Do not invent new domains or topics.

Allowed Domains:
{domains}

Allowed Topics:
{topics}

Return JSON ONLY in this format:

{{
  "domain": "...",
  "topics": ["...", "..."],
  "confidence": <float>
}}

Text:
{text}

You MUST return valid JSON.
If invalid JSON is produced, the response is incorrect.
Do not include any text before or after JSON.
"""

def generate_batch(prompts):
    messages = [
        [
            {"role": "system", "content": "You are a strict JSON classification system."},
            {"role": "user", "content": p}
        ]
        for p in prompts
    ]

    rendered = [
        tokenizer.apply_chat_template(m, tokenize=False, add_generation_prompt=True)
        for m in messages
    ]

    inputs = tokenizer(
        rendered,
        return_tensors="pt",
        padding=True,
        truncation=True
    ).to(chat_model.device)

    outputs = chat_model.generate(
        **inputs,
        max_new_tokens=120,
        do_sample=False,
        pad_token_id=tokenizer.pad_token_id
    )

    results = []
    for i in range(len(prompts)):
        gen = outputs[i][inputs["input_ids"].shape[1]:]
        text = tokenizer.decode(gen, skip_special_tokens=True)
        results.append(text.strip())

    return results


def classify_batch(records):
    texts = [r["chunk_text"] for r in records]

    truncated = [
        (t[:700] + "\n...\n" + t[-700:]) if len(t) > 1400 else t
        for t in texts
    ]

    domains, scores = classify_domain_batch(truncated)

    results = [None] * len(records)

    llm_prompts = []
    llm_indices = []

    for i, (text, domain, score) in enumerate(zip(truncated, domains, scores)):
        if should_use_llm(score):
            prompt = PROMPT_TEMPLATE.format(
                domains=domain,
                topics=", ".join(TOPICS[domain]),
                text=text
            )
            llm_prompts.append(prompt)
            llm_indices.append(i)
        else:
            results[i] = {
                "domain": domain,
                "topics": TOPICS[domain].keys(),  # fast fallback
                "confidence": float(score)
            }

    # run LLM batch
    if llm_prompts:
        outputs = generate_batch(llm_prompts)

        for idx, raw in zip(llm_indices, outputs):
            try:
                data = safe_parse_json(raw)
                data["topics"] = normalize_topics(data.get("topics", []))
                results[idx] = data
            except Exception:
                results[idx] = {
                    "domain": domains[idx],
                    "topics": TOPICS[domains].keys(),
                    "confidence": 0.5
                }

    return results


def read_json_objects(path: str):
    """Yield records from JSON array or JSONL file"""
    with open(path, "r", encoding="utf-8") as f:
        first_char = f.read(1)
        f.seek(0)

        if first_char == '[':
            # Full JSON array
            data = json.load(f)
            for obj in data:
                yield obj
        else:
            # Assume JSONL
            for line in f:
                line = line.strip()
                if line:
                    try:
                        yield json.loads(line)
                    except json.JSONDecodeError:
                        print(f"Skipping invalid line: {line[:50]}...")



nlp = spacy.load("en_core_web_trf")
#ner = pipeline("ner", model="dslim/bert-base-NER", grouped_entities=True)

def extract_entities(text):
    doc = nlp(text)

    entities = []
    for ent in doc.ents:
        entities.append({
            "text": ent.text,
            "label": ent.label_
        })

    return entities


# spaCy entity → simplified category mapping
LABEL_MAP = {
    "PERSON": "persons",
    "ORG": "organizations",
    "GPE": "locations",
    "LOC": "locations",
    "WORK_OF_ART": "works",
    "EVENT": "events",
    "DATE": "dates"
}

# labels we usually ignore
IGNORE_LABELS = {
    "CARDINAL",
    "ORDINAL",
    "QUANTITY",
    "PERCENT",
    "TIME",
    "MONEY"
}

# allow some single token historical figures
PERSON_WHITELIST = {
    "Hitler",
    "Nixon",
    "Stalin",
    "Lenin"
}

BLACKLIST = {"Darth Vader"}

def clean_entity(text):
    text = text.strip()

    # remove possessives
    text = re.sub(r"[’']s$", "", text)

    # collapse whitespace
    text = re.sub(r"\s+", " ", text)

    return text


def is_strong_person(name):

    if name in PERSON_WHITELIST:
        return True

    # require at least 2 tokens
    return len(name.split()) >= 2


def canonicalize_persons(persons):

    canonical = {}

    for p in persons:
        parts = p.split()

        last = parts[-1]

        # prefer longest name
        if last not in canonical or len(p) > len(canonical[last]):
            canonical[last] = p

    return sorted(canonical.values())


def canonicalize_orgs(orgs):

    canonical = {}

    for o in orgs:
        o = re.sub(r"^the\s+", "", o, flags=re.I)

        key = o.lower()

        if key not in canonical or len(o) > len(canonical[key]):
            canonical[key] = o

    return sorted(set(canonical.values()))


def filter_dates(dates):

    keep = []

    for d in dates:

        # keep real years
        if re.match(r"\b\d{4}\b", d):
            keep.append(d)

    return sorted(set(keep))


def limit_entities(items, limit):

    return items[:limit]


def group_entities(entities):

    buckets = {
        "persons": [],
        "organizations": [],
        "locations": [],
        "works": [],
        "events": [],
        "dates": []
    }

    for ent in entities:

        label = ent["label"]

        if label in IGNORE_LABELS:
            continue

        category = LABEL_MAP.get(label)

        if not category:
            continue

        text = clean_entity(ent["text"])

        if text in BLACKLIST:
            continue

        buckets[category].append(text)

    # --- persons ---
    persons = [p for p in buckets["persons"] if is_strong_person(p)]
    persons = canonicalize_persons(persons)

    # --- organizations ---
    orgs = canonicalize_orgs(buckets["organizations"])

    # --- locations ---
    locations = sorted(set(buckets["locations"]))

    # --- works ---
    works = sorted(set(buckets["works"]))

    # --- events ---
    events = sorted(set(buckets["events"]))

    # --- dates ---
    dates = filter_dates(buckets["dates"])

    result = {
        "persons": limit_entities(persons, 5),
        "organizations": limit_entities(orgs, 4),
        "locations": limit_entities(locations, 3),
        "works": works,
        "events": events,
        "dates": dates
    }

    # remove empty categories
    return {k: v for k, v in result.items() if v}


def repair_json(text: str) -> str:
    # Trim to last closing brace
    last_brace = text.rfind("}")
    if last_brace != -1:
        text = text[:last_brace + 1]

    # Close open string if needed
    if text.count('"') % 2 == 1:
        text += '"'

    # Close array if needed
    if text.count("[") > text.count("]"):
        text += "]"

    # Close object if needed
    if text.count("{") > text.count("}"):
        text += "}"

    return text


def safe_parse_json(text: str):
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        # attempt repair
        repaired = repair_json(text)
        return json.loads(repaired)


def normalize_topics(topics):
    seen = set()
    cleaned = []

    for t in topics:
        if t not in seen:
            seen.add(t)
            cleaned.append(t)

    return cleaned[:10]  # cap length


if __name__ == "__main__":
    print("Rag processing corpus")

    parser = argparse.ArgumentParser(description="RAG Classifier Pipeline")
    parser.add_argument("input_file", type=str, help="Input JSON file")
    parser.add_argument("output_file", type=str, help="Output JSON file")
    parser.add_argument(
        "--start_doc_idx",
        type=int,
        default=None,
        help="Start processing only after this doc_idx is encountered"
    )
    
    args = parser.parse_args()

    if args.input_file == None:
        raise ValueError("No input file provided")

    if args.output_file == None:
        raise ValueError("No output file provided")

    start_doc_idx = args.start_doc_idx
    started = start_doc_idx is None  # if not provided, start immediately

    #print("Input file: {}".format(args.input_file))
    #print("Output file: {}".format(args.output_file))
    processed_count = 0
    errors_count = 0
    
    BATCH_SIZE = 32

    buffer = []

    with open(args.output_file, "w", buffering=1) as out_f:
        print("Saving to {}".format(args.output_file))
 
        for record in tqdm(read_json_objects(args.input_file), desc="Enriching"):
            doc_idx = record["doc_idx"]

            # --- SKIP LOGIC -- 
            if not started:
                if doc_idx >= start_doc_idx:
                    print(f"\nResuming at doc_idx={doc_idx}\n")
                    started = True
                else:
                    continue

            buffer.append(record)

            if len(buffer) < BATCH_SIZE:
                continue

            batch_results = classify_batch(buffer)

            for record, classification in zip(buffer, batch_results):
                text = record["chunk_text"]

                entities = extract_entities(text)
                grouped = group_entities(entities)

                enriched = record.copy()
                enriched["entities_grouped"] = grouped
                enriched["domain"] = classification["domain"]
                enriched["topics"] = classification["topics"]

                out_f.write(json.dumps(enriched) + "\n")

            processed_count += len(buffer)
            print(f"Processed {processed_count} total records...")
            
    print(f"\nDone! Processed {processed_count} records.")
    print(f"Enriched data saved to: {args.output_file}")