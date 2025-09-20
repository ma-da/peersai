# gen_qa_offline.py
import json, sys
from pathlib import Path
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# Pick a compact text2text model that follows instructions decently.
# flan-t5-base is easy to run on CPU; upgrade to -large if you have GPU VRAM.
MODEL_ID = "google/flan-t5-base"

PROMPT = """You are a question-generation assistant.
Given the CONTEXT below, write 2-4 diverse question-answer pairs that a student could answer
from the context alone. Prefer specific, factual questions. Keep answers short (≤15 words).
Return a JSON array of objects with keys: question, answer.

CONTEXT:
{ctx}

JSON:
"""

def gen_pairs(ctx, tok, model, max_new_tokens=256, debug=False):
    inp = PROMPT.format(ctx=ctx.strip())
    ids = tok(inp, return_tensors="pt", truncation=True, max_length=1024).input_ids
    out = model.generate(
        ids,
        max_new_tokens=max_new_tokens,
        do_sample=True,
        top_p=0.9,
        temperature=0.7,
        num_return_sequences=1,
        repetition_penalty=1.1,
    )
    text = tok.decode(out[0], skip_special_tokens=True)
    print(f"text: {text}")
    # Be forgiving: try to extract a JSON array even if the model adds prose.
    start = text.find("["); end = text.rfind("]")
    if start != -1 and end != -1 and end > start:
        text = text[start:end+1]
    try:
        data = json.loads(text)
        # light validation + truncate long answers
        clean = []
        for item in data:
            q = (item.get("question") or "").strip()
            a = (item.get("answer") or "").strip()

            print(f"Q_raw: {q}")
            print(f"A_raw: {a}")

            if q and a:
                clean.append({"question": q, "answer": a})
                if debug == True:
                    print("---")
                    print(f"Q: {q}")
                    print(f"A: {a}")
        return clean[:6]
    except Exception:
        return []

def main(tok, model, infile, outfile, limit = 0):
    inpath, outpath = Path(infile), Path(outfile)
    if not inpath.is_file():
        print(f"Infile does not exist: {infile}")
        return
    with inpath.open("r", encoding="utf-8") as f, outpath.open("w", encoding="utf-8") as w:
        num_processed = 0
        for i, line in enumerate(tqdm(f, desc="Generating")):
            ctx = line.strip()
            if not ctx: continue
            pairs = gen_pairs(ctx, tok, model, 256, True)
            # Fallback: if model failed, emit a trivial Q/A
            if not pairs:
                pairs = [{"question": "What is the main point of this text?", "answer": ctx[:150]}]
            for p in pairs:
                w.write(json.dumps({
                    "id": f"line-{i}",
                    "context": ctx,
                    "question": p["question"],
                    "answer": p["answer"]
                }, ensure_ascii=False) + "\n")
            num_processed = num_processed + 1
            if 0 < limit < num_processed:  # if limit exists and num_processed exceeds it
                print(f"** Processed up to limit {limit} number of lines. Stopping. **")
                return

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python gen_qa_offline.py input.txt output.jsonl")
        sys.exit(1)

    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_ID)

    main(tok, model, sys.argv[1], sys.argv[2], 3)
