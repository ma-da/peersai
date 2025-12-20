import os
import logging
import time
import requests
import queue
from huggingface_hub import InferenceClient
from concurrent.futures import ThreadPoolExecutor

import db

# model settings
ENDPOINT_URL = "https://cr41uamktrsdyg3d.us-east-1.aws.endpoints.huggingface.cloud"

# DO NOT CHECK IN THE ACTUAL HF TOKEN
HF_TOKEN = os.environ.get('HF_TOKEN', '')

# Optional: custom warmup prompt (keeps your esoteric style)
WARMUP_PROMPT = "Q: [warmup] A:"
WARNUP_MAX_NEW_TOKENS = 16

# headers sent in endpoint model requests
REQ_HEADERS = {
    "Authorization": f"Bearer {HF_TOKEN}",
    "Content-Type": "application/json",
}

# health check payload
payload = {"inputs": "payload"}

# poll interval used by background workers
WORKER_POLL_INTERVAL_SECS = 2

# the number of workers to use in the thread pool.
# usually this should be set to max of the number of model instance replicas
MAX_WORKERS = 1

# this queue holds the jobs awaiting to be processed
job_queue = queue.Queue()

# this queue holds outgoing responses that need to be sent to client
outgoing_queue = queue.Queue()

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

    health_payload = {"inputs": "health_check"}
    try:
        r = requests.post(f"{ENDPOINT_URL}",
                          headers=REQ_HEADERS,
                          json=health_payload,
                          timeout=timeout)
        if r.status_code == 200:
            if r.json().get("health") == "ok":
                logging.info("Model ready: Custom health response received")
                return True
            else:
                logging.info("Processed response but not explicit health OK")
                return False  # Or False if strict
        elif r.status_code in (401, 403):
            logging.error(f"Auth error {r.status_code}: Invalid HF_TOKEN?")
            return False
        elif r.status_code == 503:
            logging.info("503: Model likely still loading (cold start)")
            return False
        else:
            logging.warning(f"Unexpected status: {r.status_code} - {r.text}")
            return False
    except requests.Timeout:
        logging.warning("Health check timed out (model loading?)")
        return False
    except Exception as e:
        logging.warning(f"Health check failed: {e}")
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


#def handle_queued_reqs():
#    while queue.not_empty():
#        job = queue.pop()
#        result = forward_to_model(job["prompt"])
#        deliver_result(job["user_id"], result)

def get_queued_jobs_from_db(limit: int = 1):
    """
    Fetch queued jobs in FIFO order.

    This function DOES NOT claim jobs.
    Claiming is done via mark_processing() to keep
    ownership atomic and race-safe.
    """
    with db.get_conn() as conn:
        cur = conn.execute("""
        SELECT id, user_id, prompt, status, created_at
        FROM jobs
        WHERE status = 'queued'
        ORDER BY created_at ASC
        LIMIT ?
        """, (limit,))
        rows = cur.fetchall()
        return [dict(row) for row in rows]


def bootstrap_queue_from_db():
    """
    Load all queued jobs from SQLite into the in-memory queue.
    Called ONCE at startup.
    """
    jobs = get_queued_jobs_from_db(limit=1000)  # or None for all

    for job in jobs:
        job_queue.put(job["id"])

    logging.info(f"bootstrapped {len(jobs)} queued jobs")


def process_job(job, process_fn):
    job_id = job["id"]

    try:
        logging.info(f"[thread] processing job {job_id}")
        result = process_fn(job["prompt"])
        db.mark_done(job_id, result)
        logging.info(f"[thread] completed job {job_id}")
    except Exception as e:
        logging.exception(f"[thread] job {job_id} failed")
        db.mark_failed(job_id, str(e))

def dispatcher_loop(process_fn):
    logging.info("dispatcher started")

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        while True:
            try:
                # Prevent overfilling the executor
                if executor._work_queue.qsize() >= MAX_WORKERS:
                    time.sleep(0.5)
                    continue

                jobs = get_queued_jobs(limit=1)
                if not jobs:
                    time.sleep(POLL_INTERVAL)
                    continue

                job = jobs[0]

                # Atomic DB claim
                if not mark_processing(job["id"]):
                    continue

                # Submit to thread pool
                executor.submit(process_job, job, process_fn)

            except Exception:
                logging.exception("dispatcher loop error")
                time.sleep(1)


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

