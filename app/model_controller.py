import os
import logging
import time
import requests
import queue
from huggingface_hub import InferenceClient

# model settings
ENDPOINT_URL = "https://cr41uamktrsdyg3d.us-east-1.aws.endpoints.huggingface.cloud"

# DO NOT CHECK IN THE ACTUAL HF TOKEN
HF_TOKEN = os.environ.get('HF_TOKEN', '')

# Optional: custom warmup prompt (keeps your esoteric style)
WARMUP_PROMPT = "Q: [warmup] A:"
WARNUP_MAX_NEW_TOKENS = 16

REQ_HEADERS = {
    "Authorization": f"Bearer {HF_TOKEN}",
    "Content-Type": "application/json",
}

job_queue = queue.Queue()

def send_prompt_to_model(prompt, max_new_tokens=250):
    if len(HF_TOKEN) < 1:
        raise ValueError("HF_TOKEN was invalid")

    start_time = time.perf_counter()
    logging.info(f"sending prompt (max_tokens {max_new_tokens}: {prompt}")

    client = InferenceClient(model=ENDPOINT_URL, token=HF_TOKEN)
    response = client.text_generation(prompt, max_new_tokens=max_new_tokens)
    logging.info(f"got response: {response}")
    end_time = time.perf_counter()
    elapsed_time = end_time - start_time
    logging.info(f"Operation took: {elapsed_time:.6f} seconds")

    return response


def is_model_ready(timeout=5):
    logging.info("checking health...")
    try:
        r = requests.get(f"{ENDPOINT_URL}/healthz", headers=REQ_HEADERS, timeout=timeout)
        if r.status_code == 200:
            logging.info(f"Got valid health response: {r}")
            return True
        elif r.status_code == 401:
            logging.info(f"401 Unauthorized: Check HF_TOKEN is valid for this endpoint.")
            return None
        elif r.status_code == 403:
            logging.info(f"403 Forbidden: Check HF_TOKEN is valid for this endpoint.")
            return None
        else:
            logging.warn(f"Got unknown status code: {r.status_code}")
    except Exception as e:
        logging.warn(f"Health check failed: {e}")
    return False

def send_warmup() -> bool:
    payload = {
        "inputs": WARMUP_PROMPT,
        "parameters": {
            "max_new_tokens": WARNUP_MAX_NEW_TOKENS,
            "temperature": 0.1,
            "stop_sequences": ["\n", "Q:"]
        }
    }
    try:
        r = requests.post(
            f"{ENDPOINT_URL}/generate",
            json=payload,
            headers=REQ_HEADERS,
            timeout=60
        )
        if r.status_code == 200:
            print(f"Warm-up successful! Response: {r.json().get('generated_text', '')[:100]}")
            return True
    except Exception as e:
        print(f"Warm-up request failed: {e}")
    return False

def handle_user_query(user_id, prompt):
    if is_model_ready():
        return send_prompt_to_model(prompt)

    queue.push({
        "user_id": user_id,
        "prompt": prompt,
        "timestamp": time.time()
    })

    send_warmup()

    return {
        "status": "warming",
        "message": "Model is starting. Your question is saved."
    }


def handle_queued_reqs():
    while queue.not_empty():
        job = queue.pop()
        result = forward_to_model(job["prompt"])
        deliver_result(job["user_id"], result)


def build_qwen_context(refs: dict, max_docs: int = 10) -> str:
    blocks = []

    for r in refs.get("results", [])[:max_docs]:
        blocks.append(
            f"[doc_id: {r['row_id']}]\n"
            f"Title: {r.get('title','')}\n"
            f"Source: {r.get('source','')}\n"
            f"Relevance: {r.get('score',0):.3f}\n\n"
            f"{r.get('text','')}".strip()
        )

    return "<EVIDENCE>\n" + "\n\n---\n\n".join(blocks) + "\n</EVIDENCE>"

def build_qwen_prompt(question: str, refs: dict) -> str:
    context = build_qwen_context(refs)

    return f"""You are a careful analyst.

Answer the question using ONLY the evidence provided.
If the evidence is insufficient, say so explicitly.
Do not use prior knowledge.
Cite sources by their doc_id in square brackets.

Question:
{question}

Evidence:
{context}

Answer:
"""

