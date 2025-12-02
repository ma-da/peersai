#!/usr/bin/env python3
"""
Hugging Face Inference Endpoint Warm-up & Health Checker
- Checks /healthz, /ready
- Sends a tiny warmup request
- Waits until model is fully loaded (no more cold start)
- Works with private endpoints (uses HF_TOKEN)
"""

import os
import time
import requests
import sys
from typing import Optional

# ================================================================
# CONFIGURATION — EDIT THESE
# ================================================================
ENDPOINT_URL = "https://cr41uamktrsdyg3d.us-east-1.aws.endpoints.huggingface.cloud"   # ← CHANGE THIS

# DO NOT CHECK IN ACTUAL HF_TOKEN!
HF_TOKEN = os.getenv("HF_TOKEN") or ""           # ← CHANGE or use env var

# Optional: custom warmup prompt (keeps your esoteric style)
WARMUP_PROMPT = "Q: [warmup] A:"
MAX_NEW_TOKENS = 16

# ================================================================
# NO NEED TO EDIT BELOW
# ================================================================
HEADERS = {
    "Authorization": f"Bearer {HF_TOKEN}",
    "Content-Type": "application/json",
}

def check_health() -> Optional[str]:
    print("checking health...")
    try:
        r = requests.get(f"{ENDPOINT_URL}/healthz", headers=HEADERS, timeout=10)
        if r.status_code == 200:
            print(f"Got valid health response: {r}")
            return r.json().get("status", "unknown")
        elif r.status_code == 401:
            print(f"401 Unauthorized: Check HF_TOKEN is valid for this endpoint.")
            return None
        elif r.status_code == 403:
            print(f"403 Forbidden: Check HF_TOKEN is valid for this endpoint.")
            return None
        else:
            print(f"Got unknown status code: {r.status_code}")
    except Exception as e:
        print(f"Health check failed: {e}")
    return None


def check_ready() -> Optional[str]:
    print("checking ready...")
    try:
        r = requests.get(f"{ENDPOINT_URL}/ready", headers=HEADERS, timeout=10)
        if r.status_code == 200:
            print(f"Got valid ready response: {r}")
            return r.json().get("status", "unknown")
        elif r.status_code == 401:
            print(f"401 Unauthorized: Check HF_TOKEN is valid for this endpoint.")
            return None
        elif r.status_code == 403:
            print(f"403 Forbidden: Check HF_TOKEN is valid for this endpoint.")
            return None
        else:
            print(f"Got unknown status code: {r.status_code}")
    except Exception as e:
        print(f"Ready check failed: {e}")
    return None


def send_warmup() -> bool:
    payload = {
        "inputs": WARMUP_PROMPT,
        "parameters": {
            "max_new_tokens": MAX_NEW_TOKENS,
            "temperature": 0.1,
            "stop_sequences": ["\n", "Q:"]
        }
    }
    try:
        r = requests.post(
            f"{ENDPOINT_URL}/generate",
            json=payload,
            headers=HEADERS,
            timeout=60
        )
        if r.status_code == 200:
            print(f"Warm-up successful! Response: {r.json().get('generated_text', '')[:100]}")
            return True
    except Exception as e:
        print(f"Warm-up request failed: {e}")
    return False

def main():
    print(f"Checking endpoint: {ENDPOINT_URL}\n")

    # initial check
    health = check_health()
    if health == "ready":
        print("Endpoint is healthy")
    else:
        print(f"Endpoint is not healthy, response: {health}")

    ready = check_ready()
    if ready == "ready":
        print("Endpoint is ready")
    else:
        print(f"Endpoint is not ready, response: {ready}")

    print("\nShould I send warmup (y/n)?")
    ans = input()
    if len(ans) == 0 or ans[0] != 'y':
        print("Goodbye!")
        sys.exit()

    # 1. Send warmup request immediately (triggers model load)
    print("Sending warm-up request...")
    send_warmup()

    # 2. Poll until healthy & ready
    print("\nWaiting for model to finish loading...", end="", flush=True)
    attempts = 0
    while attempts < 60:  # max ~10 minutes
        health = check_health()
        ready = check_ready()

        if health == "ready" and ready == "ready":
            print("\n\nENDPOINT IS FULLY WARM AND READY!")
            print("You can now send real queries — no cold start delay")
            return
        else:
            print(f"Endpoint is not good. Attempt {attempts+1}")

        attempts += 1
        print(".", end="", flush=True)
        time.sleep(10)

    print("\n\nTimeout: endpoint did not become ready in 10 minutes.")
    print("Check logs in HF dashboard → Settings → Logs")

if __name__ == "__main__":
    if "your-username" in ENDPOINT_URL or "hf_your_token" in HF_TOKEN:
        print("ERROR: Please update ENDPOINT_URL and HF_TOKEN in the script!")
    else:
        main()
