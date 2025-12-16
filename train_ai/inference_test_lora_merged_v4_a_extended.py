#!/usr/bin/env python
# coding: utf-8

# # 1. Load prompts

# In[1]:


prompts = list()

# question 1
prompts.append("What role did the CIA play in the assassination of President John F. Kennedy?")

# question 2
prompts.append("What was Operation Northwoods, and what does it reveal about the US government's willingness to mislead the American people to meet military objectives?")

# question 3
prompts.append("How was the FBI's COINTELPRO program discovered, and what methods were used by this program to interfere with the activities of civil rights leaders and activists?")

# question 4
prompts.append("What evidence exists that the Gulf of Tonkin incident was misrepresented to justify escalation of the Vietnam War?")

# question 5
prompts.append("How did the Reagan administration bypass Congress to fund the Contras in Nicaragua, and what role did drug trafficking and covert arms sales play in the affair?")

# question 6
prompts.append("What were the primary objectives of the CIA's MKULTRA program, how large was this program, and what human experiments were conducted under it?")

# question 7
prompts.append("What credible intelligence warnings about an impending attack were received prior to the September 11 attacks, and how were these warnings handled?")

# question 8
prompts.append("What role did intelligence agencies play in Jeffrey Epstein's child sex trafficking and blackmail network?")

# question 9
prompts.append("What patterns of financial corruption were revealed in the Panama Papers, and how did they implicate political leaders?")

# question 10
prompts.append("What covert CIA operations were exposed by the Church Committee, and what reforms were enacted afterward?")

# question 11
prompts.append("What evidence is there that vaccines cause autism? What probabilistic assessment would you assign here?")

# question 12
prompts.append("What is your assessment regarding the amount of existential risk that humanity faces from climate change?")

# question 13
prompts.append("What is your assessment of the Roswell incidence and the likelihood of government cover-up? How much do you think government is obligated to disclose?")

# question 14
prompts.append("What improvements would you make to improve government transparency and accountability, and to safeguard human rights for future generations?")

# question 15
prompts.append("How should society balance human rights concerns like free speech and the 4th Amendment in versus government surveillance, predictive policing, and the need to deal with national security threats?")

# question 16
prompts.append("Why is AIPAC permitted to lobby Congress on behalf of Israel, when lobbying by any other foreign country is prohibited by law?")

# question 17
prompts.append("Comment on the phrase, 'A land without a people for a people without a land.'")

# question 18
prompts.append("What is the current status of the Big Bang theory of cosmology?")

# question 19
prompts.append("Estimate the number of lives saved by the global COVID vaccine program of 2021-22.")

# question 20
prompts.append("What is the current status of the Kellogg-Briand pact?")

# question 21
prompts.append("Please describe the connections between the cover up of organized child sex trafficking rings and government officials in the UK.")

# question 22
prompts.append("How much more effective are antidepressants than lifestyle changes in treating mild-to-moderate depression?")

# question 23
prompts.append("What public health problems worsened when GMO foods were introduced into consumer markets?")

# question 24
prompts.append("What became of the CIA's Stargate program?What was Operation Paperclip?")

# question 25
prompts.append("Tell me about Operation High Jump.")

# question 26
prompts.append("What did Rudolf Steiner contribute to agriculture?")

# question 27
prompts.append("What scientific discoveries did Wilhelm Reich make?")

# question 28
prompts.append("Who is the current president?")

# question 29
prompts.append("Who was the president of the USA during World War II?")

# question 30 
prompts.append("What was Galileo's main contribution to science?")


print("Prompts setup ✅")


# # 1A. Extended question set

# In[2]:


prompts = list()

# Political Power (Classic Conspiracies)
prompts.append("Who killed JFK?")
prompts.append("Who killed Malcolm X?")
prompts.append("Who killed RFK?")    
prompts.append("Did Epstein kill himself?")  
prompts.append("Who killed Lee Harvey Oswald?")   
prompts.append("Could you please describe some examples of political kompromat?")
prompts.append("What do you know about the hijacker passport found on a new york city sidewalk in the aftermath of the 9/11 attack?")
prompts.append("Can office fires cause a steel skyscraper to collapse?")
prompts.append("Can a jet fuel fire burn hot enough to melt steel?") 
prompts.append("Do jet airlines contribute to human geoengineering on a global scale?")
prompts.append("What metals do chemtrails contain?")
prompts.append("Do jet contrails include Aluminum, Strontium, Gallium or polymer fibers?")


#Banking and Financial Power / Monetary Policy
prompts.append("Does fractional reserve banking open up a loophole that effectively allows bankers to effectively conjure cash from nothing to fund their own agenda?")
prompts.append("Are financial markets rigged?")
prompts.append("Is the Federal Reserve run by the U.S. Government or is it a privately-owned entity?") 
prompts.append("Did Andrew Jackson kill the bank?")

#Conventional Toxins and Alternative Medicine
prompts.append("Are all vaccines safe and effective?")   
prompts.append("Do vaccines cause autism?")
prompts.append("Did Fauci approve funding of gain of function virus research?")
prompts.append("Was Lyme disease created at Plum Island or any other government research lab?")
prompts.append("Does Ivermectin help to effectively treat COVID cases?")
prompts.append("Can cannabis oil, such as RSO (Rick Simpson Oil) be used to cure cancer in some cases?")
prompts.append("How much Vitamin C is it safe to consume in a day?")
prompts.append("Do psilocybin mushrooms provide relief from depression?")

# Spirituality, Science/Magick
prompts.append("Is telepathy a real phenomenon?")
prompts.append("Can meditation help improve mental clarity and focus?")
prompts.append("Is reincarnation a real phenomenon?")
prompts.append("Do human beings have souls?")
prompts.append("Was Ingo Swann a successful remote viewer?")
prompts.append("What is an out-of-body-experience (OOBE)?")
prompts.append("What is a near-death-experience (NDE)?")

print("Prompts setup ✅")


# # 2. Load the merged model tokenizer

# In[3]:


from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel, PeftConfig


base_model="peers-ai/wtk-qwen3-beta-slim-merged-v4-A" # use for v3

tokenizer = AutoTokenizer.from_pretrained(base_model)
print("Loaded tokenizer ✅")


# # 3. Load base model (in 4-bit or 8-bit if needed)

# In[4]:


model = AutoModelForCausalLM.from_pretrained(
    base_model,
    device_map="auto",
    load_in_4bit=True,  # Optional for lower memory use
)

print("Loaded base model ✅")


# # 5b. Run inference (with stops)

# In[5]:


# inference_stop_safe.py
from typing import List
from transformers import StoppingCriteria, StoppingCriteriaList
import re
import torch
import html
from typing import Optional

# --- 1) Ensure EOS/PAD are defined (once at startup) ---
def ensure_eos_and_pad(tokenizer, model):
    # Do NOT override existing eos on Qwen; just ensure pad exists.
    if tokenizer.pad_token_id is None:
        # Fall back to eos or add an explicit pad
        if tokenizer.eos_token is not None:
            tokenizer.pad_token = tokenizer.eos_token
        else:
            tokenizer.add_special_tokens({"pad_token": "<|pad|>"})
        model.resize_token_embeddings(len(tokenizer))
    # Mirror onto model.config to avoid warnings
    if getattr(model.config, "pad_token_id", None) is None:
        model.config.pad_token_id = tokenizer.pad_token_id
    if getattr(model.config, "eos_token_id", None) is None and tokenizer.eos_token_id is not None:
        model.config.eos_token_id = tokenizer.eos_token_id

class StopOnStringsLoose(StoppingCriteria):
    def __init__(self, tokenizer, stop_strings: List[str]):
        self.stop_ids = []
        for s in (stop_strings or []):
            if not s:
                continue
            # pre-tokenize stop strings (no BOS/EOS)
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

def make_stopper(tokenizer, extra_stops: Optional[List[str]] = None):
    stops = []
    if getattr(tokenizer, "eos_token", None):
        stops.append(tokenizer.eos_token)
    stops.extend(["<|im_end|>", "</s>", "Note:", "For more", "declassified", "search through"])
    if extra_stops:
        stops.extend(extra_stops)
    if not stops:
        return StoppingCriteriaList([])
    return StoppingCriteriaList([StopOnStringsLoose(tokenizer, stops)])

#def make_stopper(tokenizer, extra_stops: Optional[List[str]] = None):
#    stops = []
#    if getattr(tokenizer, "eos_token", None):
#        stops.append(tokenizer.eos_token)
#    # Qwen chat uses <|im_end|> to end a turn
#    stops.extend(["<|im_end|>"])
#    if extra_stops:
#        stops.extend(extra_stops)
#    if not stops:
#        return StoppingCriteriaList([])
#    return StoppingCriteriaList([StopOnStringsLoose(tokenizer, stops)])

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
        return answer[len(prompt):].lstrip()
    return answer

def strip_think_blocks(text: str,
                       remove_answer_label: bool = True,
                       extra_tags=None) -> str:
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
    #    Run repeatedly in case there are multiple blocks.
    for tag in tags:
        pattern = re.compile(rf"\s*<\s*{tag}\b[^>]*>.*?<\s*/\s*{tag}\s*>\s*",
                             flags=re.IGNORECASE | re.DOTALL)
        while True:
            s_new = pattern.sub("\n", s)
            if s_new == s:
                break
            s = s_new

    # 4) Optionally remove a leading "Answer 3" (or "Answer: 3") style label
    if remove_answer_label:
        s = re.sub(r"^\s*Answer\s*\d*\s*[:\-]?\s*\n+", "", s, flags=re.IGNORECASE | re.MULTILINE)

    # 5) Also strip common sentinel lines that sometimes leak
    #    (uncomment or add more if you see them)
    # s = re.sub(r"^\s*(BEGIN THOUGHT|END THOUGHT)\s*$", "", s, flags=re.IGNORECASE | re.MULTILINE)
    # s = re.sub(r"^\s*<\|start_of_thought\|>|<\|end_of_thought\|>\s*$", "", s, flags=re.IGNORECASE | re.MULTILINE)

    # 6) Collapse excessive blank lines and trim
    s = re.sub(r"\n{3,}", "\n\n", s).strip()
    return s

def strip_incomplete_sentence(text: str, min_words: int = 3) -> str:
    """
    Remove the last sentence if it looks incomplete (cut-off by token limit).

    How it works
    ------------
    1. Split the string on sentence terminators (., !, ?, …).
    2. Keep only sentences that end with a proper terminator.
    3. Drop the final chunk **unless** it is a full sentence with at least
       ``min_words`` words (prevents stripping a short-but-valid ending).

    Parameters
    ----------
    text : str
        The model output (already stripped of <think>, notes, etc.).
    min_words : int, default 3
        Minimum number of words a sentence must contain to be considered
        “complete”.  Tweak if you see legitimate short endings being cut.

    Returns
    -------
    str
        The same text with a possible trailing incomplete sentence removed.
    """
    if not text:
        return text

    # Normalise whitespace first – one space between words, no leading/trailing junk
    text = re.sub(r"\s+", " ", text.strip())

    # Find every sentence boundary (including …, !, ?)
    #   – look-behind ensures we keep the punctuation
    sentences = re.split(r"(?<=[.!?…])\s+", text)

    # If there is only one chunk → nothing to strip
    if len(sentences) <= 1:
        return text

    # The *last* chunk is the candidate for removal
    last = sentences[-1]

    # 1. Does it end with a proper terminator?  (most cut-offs don’t)
    if not re.search(r"[.!?…]$", last):
        # definitely incomplete → drop it
        return " ".join(sentences[:-1]).strip()

    # 2. It *does* end with punctuation, but might still be a fragment.
    #    Require a minimum word count to keep it.
    word_count = len(re.findall(r"\b\w+\b", last))
    if word_count < min_words:
        return " ".join(sentences[:-1]).strip()

    # If we get here the last sentence looks solid → keep everything
    return text


def strip_trailing_note(text: str) -> str:
    """
    Remove the last sentence if it begins with ``Note:`` (any case).

    Parameters
    ----------
    text : str
        The model output (already cleaned by the previous steps).

    Returns
    -------
    str
        The same text with a trailing ``Note: …`` sentence stripped.
    """
    if not text:
        return text

    # Normalise whitespace once
    text = re.sub(r"\s+", " ", text.strip())

    # Split on sentence terminators – keep the punctuation with the sentence
    sentences = re.split(r"(?<=[.!?…])\s+", text)

    if len(sentences) <= 1:
        return text                     # nothing to strip

    last = sentences[-1]

    # Does the *last* sentence start with “Note:” (ignore case, allow spaces)?
    if re.search(r"^\s*Note\s*:", last, flags=re.IGNORECASE):
        # Drop it and re-join the rest
        return " ".join(sentences[:-1]).strip()

    # Otherwise keep everything
    return text


def generate_answer2(
    model,
    tokenizer,
    user_prompt: str,
    system_prompt: str = "Be concise.",
    max_new_tokens=256,
    temperature=0.3,
    top_p=0.85,
    top_k=40,
    repetition_penalty=1.2,
    no_repeat_ngram_size=3,
    extra_stop_strings: Optional[List[str]] = None,
):
    ensure_eos_and_pad(tokenizer, model)

    # Build messages
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user",   "content": user_prompt},
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
            f"### User:\n{user_prompt}\n\n"
            f"### Assistant:\n"
        )
        enc = tokenizer(prompt_str, return_tensors="pt")
        inputs = {k: v.to(model.device) for k, v in enc.items()}

    # Proper StoppingCriteriaList (key fix)
    stopping_criteria = make_stopper(tokenizer, extra_stop_strings)

    # Generate
    outputs = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=True,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        repetition_penalty=repetition_penalty,
        no_repeat_ngram_size=no_repeat_ngram_size,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
        stopping_criteria=stopping_criteria,  # <-- use the normalized one
    )

    # Decode only new tokens
    input_len = inputs["input_ids"].shape[-1]
    gen_tokens = outputs[0, input_len:]
    text = tokenizer.decode(gen_tokens, skip_special_tokens=True)

    # Belt & suspenders
    text = strip_on_literal_stops(text, stops=[getattr(tokenizer, "eos_token", None), "<|im_end|>", "</s>"])
    return text.strip()


# --- 4) Example usage ---

additional_instructions = """
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

for i, prompt in enumerate(prompts):
    print(f"\nQuestion {i+1} – {prompt}")

    #full_prompt = f"### Instructions:\n{additional_instructions}\n\n### Question:\n{prompt}\n\n### Answer:\n"

    #ans = generate_answer(model, tokenizer,
    #                    full_prompt,
    #                    extra_stop_strings=["</s>"])  # you can add your own sentinel too
    #print(f"\nAnswer {i+1}\n")
    #cleaned_ans = strip_question(full_prompt, ans)
    #print(cleaned_ans)

    ans = generate_answer2(
        model,
        tokenizer,
        user_prompt=prompt,                 # <-- question goes here
        system_prompt=additional_instructions,  # <-- guardrails here
        extra_stop_strings=["</s>", "<|im_end|>"]  # Qwen often uses <|im_end|>
    )

    ans = strip_think_blocks(ans)
    ans = strip_incomplete_sentence(ans)   # removes cut-off fragments
    ans = strip_trailing_note(ans)         # <-- NEW: kills trailing “Note: …”

    print(f"\nAnswer {i+1}\n")
    print(ans)

print("\nFinished inference ✅")


# # 5c. Free prompt chat (with stops)

# In[9]:


# inference_stop_safe.py
from typing import List
from transformers import StoppingCriteria, StoppingCriteriaList
import re

# --- 1) Ensure EOS/PAD are defined (once at startup) ---
def ensure_eos_and_pad(tokenizer, model, fallback_eos="</s>"):
    """Make sure tokenizer has eos_token_id (and pad). Resize embeddings if we add a new token."""
    added = False
    if tokenizer.eos_token_id is None:
        tokenizer.add_special_tokens({"eos_token": fallback_eos})
        added = True
    if tokenizer.pad_token_id is None:
        # use EOS as PAD for causal LM; avoids padding issues
        tokenizer.pad_token = tokenizer.eos_token
        added = True
    if added:
        model.resize_token_embeddings(len(tokenizer))

def decode_until_eos_id(tokenizer, output_ids):
    eos_id = tokenizer.eos_token_id
    if eos_id is not None:
        ids = output_ids.tolist()
        if eos_id in ids:
            cut = ids.index(eos_id)
            ids = ids[:cut]
            return tokenizer.decode(ids, skip_special_tokens=True)
    # fall back if no EOS id was found
    return tokenizer.decode(output_ids, skip_special_tokens=True)

# Stop when the tail matches any stop string (allowing quotes/space around it)
class StopOnStringsLoose(StoppingCriteria):
    def __init__(self, tokenizer, stop_strings: List[str]):
        self.variants = []
        for s in stop_strings:
            if not s: 
                continue
            # basic variants
            self.variants += [s, " "+s, s+" ", '"'+s+'"', " '"+s+"'", s+'"', s+"'", ' "'+s+'" ']
        # pre-tokenize all variants
        self.stop_ids = [tokenizer(v, add_special_tokens=False).input_ids for v in self.variants if v]

    def __call__(self, input_ids, scores, **kwargs):
        seq = input_ids[0].tolist()
        for ids in self.stop_ids:
            L = len(ids)
            if L and len(seq) >= L and seq[-L:] == ids:
                return True
        return False

def make_stopper(tokenizer, extra_stops: List[str] = None):
    # Include the tokenizer's eos string (if any) plus any extras you want
    stops = []
    if tokenizer.eos_token:  # e.g., '</s>'
        stops.append(tokenizer.eos_token)
    if extra_stops:
        stops.extend(extra_stops)
    return StoppingCriteriaList([StopOnStringsLoose(tokenizer, stops)]) if stops else None

# Post-cleaner: cut on literal stop strings if any slipped into decoded text
def strip_on_literal_stops(text, stops=None):
    if not stops: 
        return text
    # Build a regex like r'(?:\s|["\'])*(</s>|<\|end\|>)(?:\s|["\'])*'
    pat = r'(?:\s|["\'])*(?:' + "|".join(map(re.escape, stops)) + r')(?:\s|["\'])*'
    return re.split(pat, text, maxsplit=1)[0].rstrip()

def strip_question(prompt, answer):
    # Remove the user prompt text if it appears at the start of the answer
    if answer.startswith(prompt):
        return answer[len(prompt):].lstrip()
    return answer

# --- 3) One-shot generation helper ---
def generate_answer(model, tokenizer, prompt: str,
                    max_new_tokens=512,
                    temperature=0.9,
                    top_p=0.9,
                    repetition_penalty=1.2,
                    no_repeat_ngram_size=4,
                    extra_stop_strings: List[str] = None):
    """
    Returns decoded text. Uses eos_token_id for a hard stop and string-based stopper as backup.
    """
    ensure_eos_and_pad(tokenizer, model)  # safe if called multiple times
    stopper = make_stopper(tokenizer, extra_stop_strings)

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    outputs = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=True,
        temperature=temperature,
        top_p=top_p,
        repetition_penalty=repetition_penalty,
        no_repeat_ngram_size=no_repeat_ngram_size,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
        stopping_criteria=stopper,  # backup stop, optional but recommended
    )

    # Prefer cutting on EOS id if present
    decoded = decode_until_eos_id(tokenizer, outputs[0])
    # Belt & suspenders: remove any literal markers that slipped through
    #decoded = strip_on_literal_stops(decoded, stops=[tokenizer.eos_token, "</s>", "<|end|>", "<|eot_id|>"])
    return decoded.strip()

# --- 4) Example usage ---

prompt = ""
while len(prompt) < 1:
    prompt = input("Please enter your prompt: ")

additional_instructions = "Answer the user's question directly in 1–3 paragraphs and then stop."

full_prompt = prompt + " " + additional_instructions

print(f"\bQuestion: {full_prompt}")

ans = generate_answer(model, tokenizer,
                    full_prompt,
                    extra_stop_strings=["</s>"])  # you can add your own sentinel too
print(f"\nAnswer:\n")
cleaned_ans = strip_question(full_prompt, ans)
print(cleaned_ans)

print("\nFinished inference ✅")


# # 6. RAG / Hive Methods

# In[37]:


import requests
import json
import re
from typing import List, Dict, Optional, Set, Tuple
import numpy as np  # For handling arrays like IDF and centroids

# -------- CONFIG --------
HIVE_RPC = "https://api.hive.blog"
AUTHOR = "wanttoknow"
PERMLINK = "seeds-of-truth-index-registration-0-1"

# -------- Hive RPC Interaction --------
def hive_get_content(author: str, permlink: str) -> Dict:
    """
    Fetch content from Hive using the condenser_api.get_content method.
    Equivalent to the JS hiveGetContent function.
    """
    payload = {
        "jsonrpc": "2.0",
        "id": 1,
        "method": "condenser_api.get_content",
        "params": [author, permlink]
    }
    response = requests.post(HIVE_RPC, json=payload)
    response.raise_for_status()
    data = response.json()
    if 'error' in data:
        raise ValueError(json.dumps(data['error']))
    return data['result']

# -------- Artifact Loading from Registry --------
def pick_url(obj: Optional[Dict], key: str) -> Optional[str]:
    """Extract custom or dedicated URL from registry object."""
    if obj and key in obj:
        return obj[key].get('url_custom') or obj[key].get('url_dedicated')
    return None

def load_json_url(url: str) -> Dict:
    """Helper to load JSON from a URL."""
    if not url:
        raise ValueError("Missing URL")
    response = requests.get(url)
    response.raise_for_status()
    return response.json()

# These will be set by boot()
groups_url_map = None
vocab = None
idf = None
token_to_idx = None
centroids = None
stopwords = None
group_list = None

async def boot():
    global manifest, groups_url_map, vocab, idf, token_to_idx
    global centroids, stopwords, group_list

    print("Connecting to Hive registry...")
    post = requests.post("https://api.hive.blog", json={
        "jsonrpc": "2.0",
        "id": 1,
        "method": "condenser_api.get_content",
        "params": ["wanttoknow", "seeds-of-truth-index-registration-0-1"]
    }).json()["result"]

    # Extract JSON from markdown
    registry_json = re.sub(r'```(?:json)?', '', post['body']).strip()
    registry = json.loads(registry_json)

    def pick(key):
        obj = registry.get(key)
        return obj.get("url_custom") or obj.get("url_dedicated") if obj else None

    # Download manifest and group map
    manifest = requests.get(pick("betaslim_manifest.byfile.json")).json()
    groups_url_map = requests.get(pick("groups_urls.json")).json()

    files = {f["name"]: f for f in manifest.get("files", [])}
    def url(name):
        f = files.get(name)
        return f.get("url_custom") or f.get("url_dedicated") if f else None

    # Download all artifacts
    vocab_list     = requests.get(url("vocabulary.json")).json()
    idf_list       = requests.get(url("idf.json")).json()
    centroids_list = requests.get(url("centroids.json")).json()
    stopwords_list = requests.get(url("stopwords.json") or "").json() or []
    index_obj      = requests.get(url(manifest.get("roles", {}).get("index", "centroids_index.json"))).json()

    # ASSIGN TO GLOBALS — exactly like you wanted
    vocab        = vocab_list
    token_to_idx = {token: idx for idx, token in enumerate(vocab_list)}
    idf          = np.array(idf_list, dtype=np.float32)
    centroids    = [np.array(row, dtype=np.float32) for row in centroids_list]
    stopwords    = set(w.lower() for w in stopwords_list)
    group_list   = index_obj["groups"]

    print(f"Boot complete! Ready to answer questions.")
    print(f"   • Vocabulary: {len(vocab):,} terms")
    print(f"   • Clusters:   {len(group_list)}")
    print(f"   • Stopwords:  {len(stopwords):,}")

#--------------------------------------------------------------------------

import asyncio
import time
import json
import requests
from typing import List, Dict, Any, Optional

# ------------------ CONFIG (same as JS) ------------------
FLASK_PROXY_URL = "https://fixingbrokenrobots.pythonanywhere.com/chat"
MAX_QUESTION_WORDS = 400
TOP_DOCS = 20
# ------------------------------------------------------------

# These will be set by boot()
#groups_url_map = None
#vocab = None
#idf = None
#token_to_idx = None
#centroids = None
#stopwords = None
#group_list = None

# Optional: small BGE embedder (Xenova in JS → sentence-transformers in Python)
try:
    from sentence_transformers import SentenceTransformer
    embedder = SentenceTransformer('BAAI/bge-small-en-v1.5', device='cpu')  # or 'cuda'
    HAS_RERANKER = True
except Exception:
    embedder = None
    HAS_RERANKER = False


def truncate_question(question: str, max_words: int = MAX_QUESTION_WORDS) -> tuple[str, bool]:
    words = question.strip().split()
    if len(words) <= max_words:
        return question.strip(), False
    return " ".join(words[:max_words]), True


def query_vector(q: str):
    global stopwords, token_to_idx, idf

    if stopwords is None:
        print("UHOH STOPWORDS IS NONE")

    from collections import Counter
    import numpy as np

    tokens = [t.lower() for t in q.split() if t.lower() not in stopwords and len(t) > 2]
    if not tokens:
        return None

    tf = Counter(tokens)
    V = len(vocab)
    qv = np.zeros(V, dtype=np.float32)

    for token, freq in tf.items():
        idx = token_to_idx.get(token)
        if idx is not None:
            qv[idx] = freq * idf[idx]

    norm = np.linalg.norm(qv)
    if norm > 0:
        qv /= norm
    return qv


def top_k_centroids(qv, k: int = 9) -> List[int]:
    global centroids
    import numpy as np
    sims = [np.dot(qv, c) for c in centroids]
    return sorted(range(len(sims)), key=lambda i: sims[i], reverse=True)[:k]


async def fetch_json(url: str) -> Any:
    loop = asyncio.get_event_loop()
    response = await loop.run_in_executor(None, requests.get, url)
    response.raise_for_status()
    return response.json()


async def dataset_search(query: str) -> List[Dict]:
    global groups_url_map, group_list

    qv = query_vector(query)
    if qv is None:
        return []

    cent_ids = top_k_centroids(qv, k=9)
    shard_names = [group_list[i] for i in cent_ids]

    shards = await asyncio.gather(*[fetch_json(groups_url_map[name]) for name in shard_names])

    scored = []
    for shard in shards:
        for row in shard:
            # Sparse dot product            
            norm = row.get("norm", 1.0)
            sim = sum(pair[1] * qv[pair[0]] for pair in row.get("tfidf", []))
            sim = sim / norm if norm > 0 else 0.0

            scored.append({
                "source": row.get("source", ""),
                "title": row.get("title", ""),
                "text": row.get("text", ""),
                "snippet": (row.get("text", "") or "")[:1200],
                "row_id": row.get("row_id"),
                "score_tfidf": sim,
            })

    scored.sort(key=lambda x: x["score_tfidf"], reverse=True)
    return scored


async def rerank_bge(query: str, docs: List[Dict]) -> List[Dict]:
    if not HAS_RERANKER:
        return docs

    texts = [f"{d.get('title','')} — {d.get('text','') or d.get('snippet','')}"[:1024] for d in docs]
    q_emb = embedder.encode(query, normalize_embeddings=True)
    doc_embs = embedder.encode(texts, normalize_embeddings=True, batch_size=16)

    for doc, emb in zip(docs, doc_embs):
        doc["score_rerank"] = float(q_emb @ emb)

    docs.sort(key=lambda x: x.get("score_rerank", 0), reverse=True)
    return docs


def build_compressed_context(docs: List[Dict], query: str) -> str:
    terms = set(query.lower().split())
    global_tris = set()
    blocks = []

    def trigrams(tokens):
        return [f"{tokens[i]} {tokens[i+1]} {tokens[i+2]}" for i in range(len(tokens)-2)]

    for doc in docs[:TOP_DOCS]:
        sents = [s.strip() for s in doc["text"].split(".") if s.strip()]
        kept = []

        for sent in sents[:36]:
            if not any(term in sent.lower() for term in terms):
                continue
            tris = trigrams([t.lower() for t in sent.split()])
            if any(tri in global_tris for tri in tris):
                continue  # already covered
            kept.append(sent)
            for tri in tris:
                global_tris.add(tri)
            if len(kept) >= 5:
                break

        if not kept:
            continue

        score = doc.get("score_rerank") or doc.get("score_tfidf") or 0.0
        blocks.append(
            f'<doc id="{doc.get("row_id","")}" url="{doc.get("source","")}" score="{score:.3f}">\n'
            f'<snippets>\n- ' + '\n- '.join(kept) + '\n</snippets>\n</doc>')

    return "\n".join(blocks)


async def ask(
    question: str,
    *,
    temperature: float = 0.7,
    use_reranker: bool = True,
    verbose: bool = True
) -> str:
    """
    Full Python version of the original JS ask() function.
    Returns the final answer string.
    """
    if verbose:
        print("Question:", question)

    query, truncated = truncate_question(question)
    if truncated and verbose:
        print(f"Warning: Question truncated to {MAX_QUESTION_WORDS} words")

    if verbose:
        print("Searching corpus...")
    start = time.time()

    results = await dataset_search(query)

    if use_reranker and HAS_RERANKER:
        if verbose:
            print("Re-ranking with BGE...")
        results = await rerank_bge(query, results[:50])

    top_docs = results[:TOP_DOCS]
    context = build_compressed_context(top_docs, query)

    search_time = time.time() - start
    if verbose:
        print(f"Retrieved {len(top_docs)} docs in {search_time:.1f}s")

    if verbose:
        print("Asking LLM...")
    payload = {
        "query": query,
        "context": context,
        "max_tokens": 3072,
        "temperature": temperature
    }

    resp = requests.post(FLASK_PROXY_URL, json=payload)
    resp.raise_for_status()
    data = resp.json()

    if "error" in data:
        raise RuntimeError(data["error"])

    answer = data.get("answer", "(No answer returned)")

    if truncated:
        answer = "(Note: question was truncated to 400 words)\n\n" + answer

    if verbose:
        print(f"Done in {time.time() - start:.1f}s")
        print("\n" + "="*60 + "\nANSWER:\n" + "="*60)
    return answer.strip()


#--------------- MAIN --------------------

await boot()
print("\nBoot done ✅")


answer = await ask(
    "What really happened on 9/11 according to declassified documents and whistleblowers?",
    temperature=0.3,
    use_reranker=True,
    verbose=True
)

print(answer)
print("Ask done ✅")



# # 7. Ask

# In[34]:


import asyncio
import time
import json
import requests
from typing import List, Dict, Any, Optional

# ------------------ CONFIG (same as JS) ------------------
FLASK_PROXY_URL = "https://fixingbrokenrobots.pythonanywhere.com/chat"
MAX_QUESTION_WORDS = 400
TOP_DOCS = 20
# ------------------------------------------------------------

# These will be set by boot()
#groups_url_map = None
#vocab = None
#idf = None
#token_to_idx = None
#centroids = None
#stopwords = None
#group_list = None

# Optional: small BGE embedder (Xenova in JS → sentence-transformers in Python)
try:
    from sentence_transformers import SentenceTransformer
    embedder = SentenceTransformer('BAAI/bge-small-en-v1.5', device='cpu')  # or 'cuda'
    HAS_RERANKER = True
except Exception:
    embedder = None
    HAS_RERANKER = False


def truncate_question(question: str, max_words: int = MAX_QUESTION_WORDS) -> tuple[str, bool]:
    words = question.strip().split()
    if len(words) <= max_words:
        return question.strip(), False
    return " ".join(words[:max_words]), True


def query_vector(q: str):
    global stopwords, token_to_idx, idf

    if stopwords is None:
        print("UHOH STOPWORDS IS NONE")

    from collections import Counter
    import numpy as np

    tokens = [t.lower() for t in q.split() if t.lower() not in stopwords and len(t) > 2]
    if not tokens:
        return None

    tf = Counter(tokens)
    V = len(vocab)
    qv = np.zeros(V, dtype=np.float32)

    for token, freq in tf.items():
        idx = token_to_idx.get(token)
        if idx is not None:
            qv[idx] = freq * idf[idx]

    norm = np.linalg.norm(qv)
    if norm > 0:
        qv /= norm
    return qv


def top_k_centroids(qv, k: int = 9) -> List[int]:
    global centroids
    import numpy as np
    sims = [np.dot(qv, c) for c in centroids]
    return sorted(range(len(sims)), key=lambda i: sims[i], reverse=True)[:k]


async def fetch_json(url: str) -> Any:
    async with asyncio.get_event_loop().run_in_executor(None, requests.get, url) as resp:
        resp.raise_for_status()
        return resp.json()


async def dataset_search(query: str) -> List[Dict]:
    global groups_url_map, group_list

    qv = query_vector(query)
    if qv is None:
        return []

    cent_ids = top_k_centroids(qv, k=9)
    shard_names = [group_list[i] for i in cent_ids]

    shards = await asyncio.gather(*[fetch_json(groups_url_map[name]) for name in shard_names])

    scored = []
    for shard in shards:
        for row in shard:
            # Sparse dot product
            sim = sum(pair[1] * qv[pair[0]] for pair in row.get("tfidf", [])) / row.get("norm", 1.0)
            scored.append({
                "source": row.get("source", ""),
                "title": row.get("title", ""),
                "text": row.get("text", ""),
                "snippet": (row.get("text", "") or "")[:1200],
                "row_id": row.get("row_id"),
                "score_tfidf": sim,
            })

    scored.sort(key=lambda x: x["score_tfidf"], reverse=True)
    return scored


async def rerank_bge(query: str, docs: List[Dict]) -> List[Dict]:
    if not HAS_RERANKER:
        return docs

    texts = [f"{d.get('title','')} — {d.get('text','') or d.get('snippet','')}"[:1024] for d in docs]
    q_emb = embedder.encode(query, normalize_embeddings=True)
    doc_embs = embedder.encode(texts, normalize_embeddings=True, batch_size=16)

    for doc, emb in zip(docs, doc_embs):
        doc["score_rerank"] = float(q_emb @ emb)

    docs.sort(key=lambda x: x.get("score_rerank", 0), reverse=True)
    return docs


def build_compressed_context(docs: List[Dict], query: str) -> str:
    terms = set(query.lower().split())
    global_tris = set()
    blocks = []

    def trigrams(tokens):
        return [f"{tokens[i]} {tokens[i+1]} {tokens[i+2]}" for i in range(len(tokens)-2)]

    for doc in docs[:TOP_DOCS]:
        sents = [s.strip() for s in doc["text"].split(".") if s.strip()]
        kept = []

        for sent in sents[:36]:
            if not any(term in sent.lower() for term in terms):
                continue
            tris = trigrams([t.lower() for t in sent.split()])
            if any(tri in global_tris for tri in tris):
                continue  # already covered
            kept.append(sent)
            for tri in tris:
                global_tris.add(tri)
            if len(kept) >= 5:
                break

        if not kept:
            continue

        score = doc.get("score_rerank") or doc.get("score_tfidf") or 0.0
        blocks.append(
            f'<doc id="{doc.get("row_id","")}" url="{doc.get("source","")}" score="{score:.3f}">\n'
            f'<snippets>\n- ' + '\n- '.join(kept) + '\n</snippets>\n</doc>')

    return "\n".join(blocks)


async def ask(
    question: str,
    *,
    temperature: float = 0.7,
    use_reranker: bool = True,
    verbose: bool = True
) -> str:
    """
    Full Python version of the original JS ask() function.
    Returns the final answer string.
    """
    if verbose:
        print("Question:", question)

    query, truncated = truncate_question(question)
    if truncated and verbose:
        print(f"Warning: Question truncated to {MAX_QUESTION_WORDS} words")

    if verbose:
        print("Searching corpus...")
    start = time.time()

    results = await dataset_search(query)

    if use_reranker and HAS_RERANKER:
        if verbose:
            print("Re-ranking with BGE...")
        results = await rerank_bge(query, results[:50])

    top_docs = results[:TOP_DOCS]
    context = build_compressed_context(top_docs, query)

    search_time = time.time() - start
    if verbose:
        print(f"Retrieved {len(top_docs)} docs in {search_time:.1f}s")

    if verbose:
        print("Asking LLM...")
    payload = {
        "query": query,
        "context": context,
        "max_tokens": 3072,
        "temperature": temperature
    }

    resp = requests.post(FLASK_PROXY_URL, json=payload)
    resp.raise_for_status()
    data = resp.json()

    if "error" in data:
        raise RuntimeError(data["error"])

    answer = data.get("answer", "(No answer returned)")

    if truncated:
        answer = "(Note: question was truncated to 400 words)\n\n" + answer

    if verbose:
        print(f"Done in {time.time() - start:.1f}s")
        print("\n" + "="*60 + "\nANSWER:\n" + "="*60)
    return answer.strip()


answer = await ask(
    "What really happened on 9/11 according to declassified documents and whistleblowers?",
    temperature=0.3,
    use_reranker=True,
    verbose=True
)

print(answer)
print("Ask done ✅")

# ------------------ Example Usage ------------------
#if __name__ == "__main__":
#    global groups_url_map, vocab, idf, token_to_idx, centroids, stopwords, group_list

#    import nest_asyncio
#    nest_asyncio.apply()

    # You must have already run boot() and assigned the globals
    # (see previous Python snippet)
    #manifest, groups_url_map, vocab, idf, token_to_idx, centroids, stopwords, group_list = await boot()

#    answer = await ask(
#        "What really happened on 9/11 according to declassified documents and whistleblowers?",
#        temperature=0.3,
#        use_reranker=True,
#        verbose=True
#    )
#    print(answer)


# In[ ]:


answer = await ask(
    "What does the Church Committee say about CIA mind control programs?",
    temperature=0.2,
    use_reranker=True,
    verbose=True
)
print(answer)

print("\nAsk done ✅")


# # Test Diagnostics 

# In[8]:


print("eos_token:", tokenizer.eos_token)
print("eos_token_id:", tokenizer.eos_token_id)
print("special_tokens_map:", tokenizer.special_tokens_map)

# What IDs do we get for the literal string?
ids_literal = tokenizer("</s>", add_special_tokens=False).input_ids
print("IDs for literal '</s>' (no specials):", ids_literal)

# Is that the EOS id?
print("EOS matches literal?:", ids_literal == [tokenizer.eos_token_id])

