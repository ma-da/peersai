# handler.py — HF Inference Endpoint for Qwen3-32B LoRA (esoteric fine-tuned)
# Supports: messages[], inputs, additional_instructions, /no_think, fast inference

import json
import logging
from typing import Any, Dict, List, Optional

import torch
import html
import re
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    pipeline,
    BitsAndBytesConfig,
)

# ----------------------------------------------------------------------
# 1. LOGGING & CONFIG
# ----------------------------------------------------------------------
logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

DEFAULT_SYSTEM = """
Answer the question in 1–3 concise paragraphs (total <300 words).
Use proper spelling, punctuation, and spacing.
Do not run words together.
Avoid long strings of numbers.
NO LISTS. If unavoidable, then limit lists to 3 items maximum.

Focus only on the question asked, avoiding unrelated topics or meta-text (e.g., "Note:", "click here").
Do not suggest other references, "further reading", or "Note:" for the reader to explore, view, or to learn more.
Stop after the answer.
/no_think
"""

# Optional: 4-bit quantization (saves VRAM, speeds up ~2x)
USE_4BIT = False
quant_config = (
    BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    if USE_4BIT
    else None
)


# ----------------------------------------------------------------------
# 1a. TEXT HELPERS
# ----------------------------------------------------------------------
def strip_on_literal_stops(text, stops=None):
    if not text or not stops:
        return text
    cut = len(text)
    for s in stops:
        if not s:
            continue
        idx = text.find(s)
        if idx != -1:
            cut = min(cut, idx)
    return text[:cut].rstrip()


def strip_question(prompt, answer):
    # Remove the user prompt text if it appears at the start of the answer
    if answer.startswith(prompt):
        return answer[len(prompt) :].lstrip()
    return answer


def strip_think_blocks(text: str, remove_answer_label: bool = True, extra_tags=None) -> str:
    """
    Remove hidden-reasoning sections like <think>...</think> (and variants),
    plus an optional leading 'Answer N' label. Also normalizes blank lines.
    """
    if not text:
        return text

    # 1) Unescape in case tags came through as &lt;think&gt;
    s = html.unescape(text)

    # 2) Build a tag list: <think>, <scratchpad>, <inner_monologue>, <reasoning>, <notes>
    tags = ["think", "scratchpad", "inner_monologue", "reasoning", "notes"]
    if extra_tags:
        tags.extend([t for t in extra_tags if t and t not in tags])

    # 3) Remove all tagged blocks, non-greedy, across newlines, case-insensitive
    #    Use a pattern that doesn't greedily eat surrounding whitespace.
    for tag in tags:
        pattern = re.compile(
            rf"<\s*{tag}\b[^>]*>.*?<\s*/\s*{tag}\s*>",
            flags=re.IGNORECASE | re.DOTALL,
        )
        while True:
            s_new = pattern.sub("\n", s)
            if s_new == s:
                break
            s = s_new

    # 4) Optionally remove a leading "Answer 3" (or "Answer: 3") style label
    if remove_answer_label:
        s = re.sub(
            r"^\s*Answer\s*\d*\s*[:\-]?\s*\n+",
            "",
            s,
            flags=re.IGNORECASE | re.MULTILINE,
        )

    # 5) Collapse excessive blank lines and trim
    s = re.sub(r"\n{3,}", "\n\n", s).strip()
    return s


def strip_incomplete_sentence(text: str, min_words: int = 3) -> str:
    """
    Remove the last sentence if it looks incomplete (cut-off by token limit).
    """
    if not text:
        return text

    text = re.sub(r"\s+", " ", text.strip())
    sentences = re.split(r"(?<=[.!?…])\s+", text)

    if len(sentences) <= 1:
        return text

    last = sentences[-1]

    if not re.search(r"[.!?…]$", last):
        return " ".join(sentences[:-1]).strip()

    word_count = len(re.findall(r"\b\w+\b", last))
    if word_count < min_words:
        return " ".join(sentences[:-1]).strip()

    return text


def strip_trailing_note(text: str) -> str:
    """
    Remove the last sentence if it begins with ``Note:`` (any case).
    """
    if not text:
        return text

    text = re.sub(r"\s+", " ", text.strip())
    sentences = re.split(r"(?<=[.!?…])\s+", text)

    if len(sentences) <= 1:
        return text

    last = sentences[-1]

    if re.search(r"^\s*Note\s*:", last, flags=re.IGNORECASE):
        return " ".join(sentences[:-1]).strip()

    return text


def extract_answer_block(text: str) -> str:
    """
    Prefer an <answer>...</answer> block if present.
    If not found, return the original text (to allow fallback logic).
    """
    if not text:
        return text
    m = re.search(
        r"<\s*answer\b[^>]*>(.*?)<\s*/\s*answer\s*>",
        text,
        flags=re.IGNORECASE | re.DOTALL,
    )
    if m:
        return m.group(1).strip()
    return text.strip()


# ----------------------------------------------------------------------
# 2. ENDPOINT HANDLER
# ----------------------------------------------------------------------
class EndpointHandler:
    def __init__(self, model_dir: str):
        log.info("Loading tokenizer and model...")
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_dir, trust_remote_code=True
        )

        model_kwargs = {
            "torch_dtype": torch.bfloat16,
            "device_map": "auto",
            "trust_remote_code": True,
        }
        if USE_4BIT and quant_config is not None:
            model_kwargs["quantization_config"] = quant_config

        self.model = AutoModelForCausalLM.from_pretrained(model_dir, **model_kwargs)

        # Fix EOS/PAD
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model.config.pad_token_id = self.tokenizer.pad_token_id
        self.model.config.eos_token_id = self.tokenizer.eos_token_id

        # Pipeline (kept for simplicity, but with better defaults)
        self.pipe = pipeline(
            task="text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            return_full_text=False,
        )
        log.info("Model loaded successfully.")

    # ------------------------------------------------------------------
    def _build_prompt(self, data: Dict[str, Any]) -> str:
        """
        Build prompt using Qwen chat template.

        Supports:
        - HF-style payload: {"inputs": {"messages": [...]}, "parameters": {...}}
        - Fallback: {"messages": [...]} or {"inputs": "raw string"}
        """
        payload = data.get("inputs")
        messages = None

        # Case 1: HF-style nested messages: inputs = {"messages": [...]}
        if isinstance(payload, dict) and "messages" in payload:
            messages = payload["messages"]

        # Case 2: direct messages at top level (e.g. local testing)
        if messages is None and isinstance(data.get("messages"), list):
            messages = data["messages"]

        # Case 3: raw string input → wrap in system+user messages
        if messages is None:
            user_input = payload or data.get("input") or ""
            messages = [
                {"role": "system", "content": DEFAULT_SYSTEM},
                {"role": "user", "content": user_input},
            ]
        else:
            # ensure there's a system message
            has_system = any(m.get("role") == "system" for m in messages)
            if not has_system:
                messages = [{"role": "system", "content": DEFAULT_SYSTEM}] + messages

        return self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

    # ------------------------------------------------------------------
    def _get_stop_ids(self, stop_list: List[str]) -> Optional[List[int]]:
        if not stop_list:
            return None
        ids = []
        for s in stop_list:
            toks = self.tokenizer.encode(s, add_special_tokens=False)
            if len(toks) == 1:
                ids.append(toks[0])
        return list(set(ids)) if ids else None

    # ------------------------------------------------------------------
    def __call__(self, data: Dict[str, Any]) -> Dict[str, Any]:
        log.info(f"Request: {json.dumps(data)[:200]}...")

        params = data.get("parameters") or {}
        max_new_tokens = int(params.get("max_new_tokens", 512))
        temperature = float(params.get("temperature", 0.3))
        top_p = float(params.get("top_p", 0.9))
        top_k = int(params.get("top_k", 40))
        repetition_penalty = float(params.get("repetition_penalty", 1.2))
        stop_list = params.get("stop", []) or []
        if isinstance(stop_list, str):
            stop_list = [stop_list]

        # Build prompt with instructions
        prompt = self._build_prompt(data)
        log.info(f"Prompt length: {len(prompt)} chars")

        # Generation kwargs
        gen_kwargs = {
            "max_new_tokens": max_new_tokens,
            "do_sample": True,
            "temperature": temperature,
            "top_p": top_p,
            "top_k": top_k,
            "repetition_penalty": repetition_penalty,
            "pad_token_id": self.tokenizer.pad_token_id,
        }

        try:
            log.info("Starting generation...")
            out = self.pipe(prompt, **gen_kwargs)
            raw_text = out[0]["generated_text"]
            log.info(f"RAW GENERATED: {repr(raw_text)[:400]}")
        except Exception as e:
            log.error(f"Generation failed: {e}")
            torch.cuda.empty_cache()
            return {"generated_text": "[Error: Generation failed]"}

        # 1) Prefer explicit <answer> block if present
        text = extract_answer_block(raw_text)

        # 2) Strip remaining <think> and clean up
        #    Combine default stops with user-provided stops
        base_stops = [
            getattr(self.tokenizer, "eos_token", None),
            "<|im_end|>",
            "</s>",
        ]
        all_stops = base_stops + stop_list

        text_clean = strip_on_literal_stops(text, stops=all_stops)
        text_clean = strip_think_blocks(text_clean)
        text_clean = strip_incomplete_sentence(text_clean)
        text_clean = strip_trailing_note(text_clean)

        # 3) Fallback: if we stripped everything, fall back progressively
        if not text_clean.strip():
            log.warning(
                "All text stripped in post-processing; falling back to raw output."
            )
            fallback = strip_think_blocks(
                strip_on_literal_stops(raw_text, stops=all_stops)
            )
            text_clean = fallback if fallback.strip() else raw_text

        torch.cuda.empty_cache()
        log.info(f"Response length: {len(text_clean)} chars")
        return {"generated_text": text_clean}
