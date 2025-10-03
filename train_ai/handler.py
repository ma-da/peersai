# handler.py — HF Inference Endpoint with safer stopping & cleanup
# Works with Qwen/DeepSeek-style chat templates and standard causal LMs.

import json
import html
import re
from typing import Any, Dict, List, Optional

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    StoppingCriteria,
    StoppingCriteriaList,
)

# ----------------------------
# Helpers borrowed & adapted from inference_stop_safe.py
# ----------------------------

def ensure_eos_and_pad(tokenizer, model):
    """
    Make sure pad/eos ids are available to avoid warnings and odd generation.
    We do *not* override an existing eos; if no pad is set, mirror eos or add a pad.
    Safe to call multiple times.
    """
    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token is not None:
            tokenizer.pad_token = tokenizer.eos_token
        else:
            tokenizer.add_special_tokens({"pad_token": "<|pad|>"})
        model.resize_token_embeddings(len(tokenizer))

    if getattr(model.config, "pad_token_id", None) is None:
        model.config.pad_token_id = tokenizer.pad_token_id
    if getattr(model.config, "eos_token_id", None) is None and tokenizer.eos_token_id is not None:
        model.config.eos_token_id = tokenizer.eos_token_id


class StopOnStringsLoose(StoppingCriteria):
    """
    Stops when the sequence ends with any *tokenized* stop string.
    """
    def __init__(self, tokenizer, stop_strings: List[str]):
        self.stop_ids = []
        self.tokenizer = tokenizer
        for s in (stop_strings or []):
            if not s:
                continue
            ids = tokenizer.encode(s, add_special_tokens=False)
            if ids:
                self.stop_ids.append(ids)

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        seq = input_ids[0].tolist()
        for ids in self.stop_ids:
            L = len(ids)
            if L and len(seq) >= L and seq[-L:] == ids:
                return True
        return False


def make_stopper(tokenizer, extra_stops: Optional[List[str]] = None) -> StoppingCriteriaList:
    stops: List[str] = []
    if getattr(tokenizer, "eos_token", None):
        stops.append(tokenizer.eos_token)
    # Qwen-style chat turn end token
    stops.extend(["<|im_end|>"])
    if extra_stops:
        stops.extend([s for s in extra_stops if s])
    if not stops:
        return StoppingCriteriaList([])
    return StoppingCriteriaList([StopOnStringsLoose(tokenizer, stops)])


def strip_on_literal_stops(text: str, stops: Optional[List[str]] = None) -> str:
    if not text or not stops:
        return text
    cut = len(text)
    for s in stops:
        if not s:
            continue
        i = text.find(s)
        if i != -1:
            cut = min(cut, i)
    return text[:cut].rstrip()


def strip_think_blocks(text: str,
                       remove_answer_label: bool = True,
                       extra_tags: Optional[List[str]] = None) -> str:
    """
    Remove <think>/<scratchpad>/<inner_monologue>/<reasoning>/<notes> ... blocks,
    collapse blank lines, and optionally drop a leading "Answer N:" label.
    """
    if not text:
        return text

    s = html.unescape(text)
    tags = ["think", "scratchpad", "inner_monologue", "reasoning", "notes"]
    if extra_tags:
        for t in extra_tags:
            if t and t not in tags:
                tags.append(t)

    for tag in tags:
        pattern = re.compile(rf"\s*<\s*{tag}\b[^>]*>.*?<\s*/\s*{tag}\s*>\s*",
                             flags=re.IGNORECASE | re.DOTALL)
        while True:
            s_new = pattern.sub("\n", s)
            if s_new == s:
                break
            s = s_new

    if remove_answer_label:
        s = re.sub(r"^\s*Answer\s*\d*\s*[:\-]?\s*\n+", "", s, flags=re.IGNORECASE | re.MULTILINE)

    s = re.sub(r"\n{3,}", "\n\n", s).strip()
    return s


def _chat_apply_or_fallback(tokenizer, messages: List[dict]) -> Dict[str, torch.Tensor]:
    """
    Apply chat template if available; otherwise build a simple prompt.
    Returns a dict suitable for model.generate(**inputs).
    """
    has_chat_template = getattr(tokenizer, "chat_template", None) not in (None, "")
    if has_chat_template:
        enc = tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
        )
        if isinstance(enc, torch.Tensor):
            input_ids = enc
            attention_mask = torch.ones_like(input_ids)
            return {"input_ids": input_ids, "attention_mask": attention_mask}
        # enc may already be a dict with input_ids/attention_mask
        return enc

    # Fallback prompt
    sys_text = ""
    user_text = ""
    for m in messages:
        if m.get("role") == "system":
            sys_text += (m.get("content") or "") + "\n"
        elif m.get("role") == "user":
            user_text += (m.get("content") or "") + "\n"
    prompt = f"### System:\n{sys_text.strip()}\n\n### User:\n{user_text.strip()}\n\n### Assistant:\n"
    return tokenizer(prompt, return_tensors="pt")


def _encode_single(tokenizer, prompt: str) -> Dict[str, torch.Tensor]:
    enc = tokenizer(prompt, return_tensors="pt")
    return enc


def _single_token_eos_ids(tokenizer, stop_list: Optional[List[str]]) -> Optional[List[int]]:
    """
    Map any stop string that is exactly one token to eos_token_ids list.
    (We still also use string-based StoppingCriteria for multi-token stops.)
    """
    if not stop_list:
        return None
    ids = []
    for s in stop_list:
        if not s:
            continue
        toks = tokenizer.encode(s, add_special_tokens=False)
        if len(toks) == 1:
            ids.append(toks[0])
    return list(sorted(set(ids))) or None


# ----------------------------
# Hugging Face Inference Endpoint
# ----------------------------

class EndpointHandler:
    def __init__(self, model_dir: str):
        """
        Loads model + tokenizer once at container startup.
        """
        torch_dtype = torch.bfloat16  # good default for modern LLMs
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_dir,
            torch_dtype=torch_dtype,
            device_map="auto",
            trust_remote_code=True,
        )
        ensure_eos_and_pad(self.tokenizer, self.model)  # safe id wiring

        # Convenience: common literal stops we’ll trim after decode
        self.default_literal_stops = [
            getattr(self.tokenizer, "eos_token", None),
            "</s>",
            "<|im_end|>",
            "<|end|>",
            "<|eot_id|>",
        ]

    def _build_inputs(self, data: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        """
        Support either chat-style {"messages":[...]} or raw {"inputs": "..."}.
        """
        if isinstance(data.get("messages"), list):
            return _chat_apply_or_fallback(self.tokenizer, data["messages"])
        prompt = data.get("inputs") or data.get("input") or ""
        return _encode_single(self.tokenizer, prompt)

    def __call__(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Per-request entry point for HF Inference Endpoints.
        Accepts:
          - {"messages": [...], "parameters": {...}}
          - {"inputs": "prompt text", "parameters": {...}}
        """
        params = data.get("parameters") or {}

        max_new_tokens       = int(params.get("max_new_tokens", params.get("max_tokens", 256)))
        temperature          = float(params.get("temperature", 0.3))
        top_p                = float(params.get("top_p", 0.9))
        repetition_penalty   = float(params.get("repetition_penalty", 1.1))
        no_repeat_ngram_size = int(params.get("no_repeat_ngram_size", 3))

        # Stop sequences from either "stop" or "stop_sequences"
        user_stops = params.get("stop") or params.get("stop_sequences") or data.get("stop") or []
        if isinstance(user_stops, str):
            user_stops = [user_stops]

        # Inputs
        inputs = self._build_inputs(data)
        # Move to model device
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
        input_len = inputs["input_ids"].shape[-1]

        # Stopping criteria (string-based, handles multi-token)
        stopping_criteria = make_stopper(self.tokenizer, user_stops)

        # Also allow any one-token stop to act as eos (fast, native cutoff)
        eos_token_ids = _single_token_eos_ids(self.tokenizer, user_stops)

        gen_kwargs = dict(
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=temperature,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            no_repeat_ngram_size=no_repeat_ngram_size,
            eos_token_id=eos_token_ids if eos_token_ids else self.tokenizer.eos_token_id,
            pad_token_id=self.tokenizer.pad_token_id,
            stopping_criteria=stopping_criteria,
        )

        with torch.no_grad():
            outputs = self.model.generate(**inputs, **gen_kwargs)

        # Decode *only* the newly generated tokens
        gen_tokens = outputs[0, input_len:]
        text = self.tokenizer.decode(gen_tokens, skip_special_tokens=True)

        # Post-process: literal stop trims + hidden-reasoning removal
        text = strip_on_literal_stops(text, stops=[s for s in (self.default_literal_stops + (user_stops or [])) if s])
        text = strip_think_blocks(text)

        # Shallow “finish_reason” best-effort
        finish_reason = "length" if gen_tokens.shape[-1] >= max_new_tokens else "stop"

        # (Optional) include token counts
        usage = {
            "prompt_tokens": int(input_len),
            "completion_tokens": int(gen_tokens.shape[-1]),
            "total_tokens": int(input_len + gen_tokens.shape[-1]),
        }

        return {
            "generated_text": text.strip(),
        }
