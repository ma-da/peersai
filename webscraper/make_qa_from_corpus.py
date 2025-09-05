#!/usr/bin/env python3
import argparse, json, os, re, pathlib
from typing import List, Tuple, Iterable, Optional

# --------- simple cleaner to drop boilerplate you saw earlier ----------
BAD_PATTERNS = [
    r"^if you wish[,:\- ]",
    r"^if you want more detail",
    r"^related articles\b",
    r"\bclick here\b",
    r"\bwatch (an?|the)?\s*video on youtube\b",
    r"^note:\b",
    r"\|$",
]
BAD_RX = [re.compile(p, re.I) for p in BAD_PATTERNS]

def clean_text(s: str) -> str:
    lines = []
    for ln in s.splitlines():
        if any(rx.search(ln.strip()) for rx in BAD_RX):
            continue
        lines.append(ln)
    s = "\n".join(lines)
    s = re.sub(r"[ \t\xa0]+", " ", s)
    s = re.sub(r"\n{3,}", "\n\n", s)
    return s.strip()

# ------------------------ input helpers ------------------------
def read_folder(folder: str) -> List[Tuple[str, str]]:
    """Each .txt file: first line = title (Q), rest = body (A)."""
    pairs = []
    for fn in sorted(os.listdir(folder)):
        if not fn.lower().endswith(".txt"):
            continue
        p = os.path.join(folder, fn)
        txt = pathlib.Path(p).read_text(encoding="utf-8", errors="ignore")
        lines = [l for l in txt.splitlines()]
        if not lines:
            continue
        title = lines[0].strip()
        body = "\n".join(lines[1:]).strip()
        if not title or not body:
            # fallback: use filename as title if needed
            if not title:
                title = pathlib.Path(fn).stem.replace("_"," ").replace("-"," ").strip()
        pairs.append((title, body))
    return pairs

def iter_blocks(lines: Iterable[str]) -> Iterable[List[str]]:
    """Yield blocks separated by blank lines."""
    buf = []
    for ln in lines:
        if ln.strip() == "":
            if buf:
                yield buf
                buf = []
        else:
            buf.append(ln.rstrip("\n"))
    if buf:
        yield buf

def read_monolith(file_path: str) -> List[Tuple[str, str]]:
    """First non-empty line of each block = title; rest = body."""
    txt = pathlib.Path(file_path).read_text(encoding="utf-8", errors="ignore")
    pairs = []
    for block in iter_blocks(txt.splitlines()):
        if not block:
            continue
        title = block[0].strip()
        body = "\n".join(block[1:]).strip()
        if title and body:
            pairs.append((title, body))
    return pairs

def read_one_per_line(file_path: str, question_template: str) -> List[Tuple[str, str]]:
    """Each line is an article; synthesize a question from the template."""
    pairs = []
    with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
        for ln in f:
            body = ln.strip()
            if not body:
                continue
            # heuristically derive a title for the question (optional)
            first_sentence = re.split(r"(?<=[.!?])\s+", body)[0][:120]
            q = question_template.format(title_hint=first_sentence)
            pairs.append((q, body))
    return pairs

# ------------------------ writer ------------------------
def to_messages(question: str, answer: str) -> dict:
    return {
        "messages": [
            {"role": "user", "content": question},
            {"role": "assistant", "content": answer},
        ],
        # Also provide a flat text version you can use with formatting_func
        "text": f"<|user|>\n{question}\n<|assistant|>\n{answer}"
    }

def save_jsonl(pairs: List[Tuple[str, str]], out_path: str, max_chars_answer: Optional[int] = None):
    n = 0
    with open(out_path, "w", encoding="utf-8") as f:
        for q, a in pairs:
            q, a = clean_text(q), clean_text(a)
            if not q or not a:
                continue
            if max_chars_answer:
                a = a[:max_chars_answer].rstrip()
            json.dump(to_messages(q, a), f, ensure_ascii=False)
            f.write("\n")
            n += 1
    print(f"✅ Wrote {n} Q/A pairs to {out_path}")

# ------------------------ main ------------------------
def main():
    ap = argparse.ArgumentParser("Build Q&A pairs from article corpus")
    mode = ap.add_mutually_exclusive_group(required=True)
    mode.add_argument("--from-folder", help="Folder of .txt files (first line = title, rest = article)")
    mode.add_argument("--from-monolith", help="Single .txt with blocks separated by blank lines")
    mode.add_argument("--from-lines", help="Single .txt with one article per line")
    ap.add_argument("--out", required=True, help="Output JSONL path (chat-style messages + text field)")
    ap.add_argument("--truncate_answer_chars", type=int, default=0, help="Optional: limit answer length in chars")
    ap.add_argument("--question_template",
                    default="Please summarize the following article in 1–3 paragraphs: {title_hint}",
                    help="Used only with --from-lines; {title_hint} is substituted.")
    args = ap.parse_args()

    if args.from_folder:
        pairs = read_folder(args.from_folder)
    elif args.from_monolith:
        pairs = read_monolith(args.from_monolith)
    else:
        pairs = read_one_per_line(args.from_lines, args.question_template)

    # optional truncation (light safety)
    max_chars = args.truncate_answer_chars or None
    save_jsonl(pairs, args.out, max_chars)

if __name__ == "__main__":
    main()

