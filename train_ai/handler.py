# handler.py works with peers-ai/deepseek-32b-my-lora1-with-stops-merged model on Hugging Face
# Hugging Face Inference Endpoints (Custom runtime) expects a class named `EndpointHandler`
# that it can instantiate as EndpointHandler(model_dir) and then call like a function.

import json
from typing import Any, Dict
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

class EndpointHandler:
    def __init__(self, model_dir: str):
        """
        Loads the model and tokenizer once at container startup.
        """
        # Match your merged config
        torch_dtype = torch.bfloat16

        self.tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_dir,
            torch_dtype=torch_dtype,
            device_map="auto",
            trust_remote_code=True,
        )

        # Standard text-generation pipeline
        self.pipe = pipeline(
            task="text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
            torch_dtype=torch_dtype,
            device_map="auto",
            return_full_text=False,  # return only the generated continuation
        )

    def _build_prompt_from_messages(self, messages):
        # Use model's chat template (Qwen/DeepSeek style)
        return self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

    def _single_token_eos_ids(self, stop_list):
        """
        Convert single-token stop strings into eos_token_id(s).
        Multi-token stops are trimmed post-generation.
        """
        if not stop_list:
            return None
        ids = []
        for s in stop_list:
            toks = self.tokenizer.encode(s, add_special_tokens=False)
            if len(toks) == 1:
                ids.append(toks[0])
        return list(set(ids)) or None

    def __call__(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Called per request. Accepts either:
          - {"messages": [...], "parameters": {...}}
          - {"inputs": "raw prompt", "parameters": {...}}
        """
        params = data.get("parameters") or {}
        max_new_tokens = int(params.get("max_new_tokens", params.get("max_tokens", 256)))
        temperature = float(params.get("temperature", 0.7))
        top_p = float(params.get("top_p", 0.9))
        stop_list = params.get("stop", data.get("stop", [])) or []

        # Build prompt
        if "messages" in data and isinstance(data["messages"], list):
            prompt = self._build_prompt_from_messages(data["messages"])
        else:
            prompt = data.get("inputs") or data.get("input") or ""

        eos_token_id = self._single_token_eos_ids(stop_list)

        gen_kwargs = dict(
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=temperature,
            top_p=top_p,
            eos_token_id=eos_token_id,
        )

        # Generate
        out = self.pipe(prompt, **gen_kwargs)
        text = out[0]["generated_text"]

        # Post-trim for multi-token stop strings
        for s in stop_list:
            i = text.find(s)
            if i != -1:
                text = text[:i]

        # Generic response shape
        return {"generated_text": text}
