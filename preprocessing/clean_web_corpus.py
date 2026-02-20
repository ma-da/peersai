#!/usr/bin/env python3
"""
clean_web_corpus.py

Cleans a directory of .txt files (webscraped / PDF-converted text) and writes
cleaned versions to an output directory with "_cleaned" appended to filenames.

Features included:
- Ignores files ending with "_cleaned.txt"
- Dry-run mode (no writes)
- strip_pre / strip_post: strip 0–25% of blocks from start/end (block = paragraph-ish chunk)
- Removes everything after the exact marker: "**What you can do:**"
- Removes markdown bold markers (**...**) while preserving content
- Normalizes markdown links [text](url) -> text
- Removes bare URLs (http/https)
- Removes TOC-like and index-like blocks
- Separates references blocks (optional; currently excluded from cleaned output)
"""

import argparse
import re
from pathlib import Path
from collections import Counter


# ----------------------------
# Text utilities
# ----------------------------

URL_RE = re.compile(r"https?://\S+")
MD_LINK_RE = re.compile(r"\[([^\]]+)\]\((https?://[^)]+)\)")
BOLD_RE = re.compile(r"\*\*(.*?)\*\*", flags=re.DOTALL)

# Marker: delete everything after this token occurs (first occurrence).
CUT_MARKER = "**What you can do:**"

# TOC-like line (common in PDFs and some scraped docs)
TOC_LINE_RE = re.compile(r"^\s*\d+(\.\d+)*\s+.*\.{3,}\s*\d+\s*$")

REFERENCE_HEADERS = {
    "references",
    "bibliography",
    "works cited",
    "literature cited",
}

META_CONTAINS_PATTERNS = [
    re.compile(r"WantToKnow\.info"),
    re.compile(r"\bPEERS\b"),
    re.compile(r"click here", re.IGNORECASE),
]

META_START_PATTERNS = [
    re.compile(r"^\s*note:", re.IGNORECASE),
    re.compile(r"^\s*for more information", re.IGNORECASE),
]


def normalize_newlines(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    # collapse extreme blank lines
    text = re.sub(r"\n{4,}", "\n\n\n", text)
    return text


def cut_after_marker(text: str, marker: str = CUT_MARKER) -> str:
    """
    Delete everything after the first occurrence of marker (including marker line).
    The user asked specifically for the token "**What you can do:**".
    """
    idx = text.find(marker)
    if idx == -1:
        return text
    return text[:idx].rstrip() + "\n"


def replace_markdown_links(text: str) -> str:
    # [label](https://...) -> label
    return MD_LINK_RE.sub(r"\1", text)


def strip_bold_markers(text: str) -> str:
    # **Something** -> Something
    return BOLD_RE.sub(r"\1", text)


def strip_urls(text: str) -> str:
    # Remove bare URLs (after markdown links already normalized)
    return URL_RE.sub("", text)


def cleanup_whitespace(text: str) -> str:
    # Strip trailing spaces and tidy repeated spaces
    text = "\n".join(line.rstrip() for line in text.splitlines())
    text = re.sub(r"[ \t]{2,}", " ", text)
    # Reduce space-only lines to empty
    text = re.sub(r"^\s+$", "", text, flags=re.MULTILINE)
    # Collapse too many blank lines
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip() + "\n"


def remove_meta_lines(text: str) -> str:
    """
    Removes lines that:
      - contain promotional/self-referential terms
      - start with certain meta prefixes
    """
    cleaned_lines = []

    for line in text.splitlines():
        stripped = line.strip()

        # Remove lines containing promotional terms
        if any(p.search(line) for p in META_CONTAINS_PATTERNS):
            continue

        # Remove lines starting with meta prefixes
        if any(p.match(stripped) for p in META_START_PATTERNS):
            continue

        cleaned_lines.append(line)

    return "\n".join(cleaned_lines)


def normalize_paragraph_linebreaks(text: str) -> str:
    """
    Converts wrapped lines inside paragraphs into single-line paragraphs.
    Preserves blank-line paragraph boundaries.
    """

    # Normalize newlines first
    text = text.replace("\r\n", "\n").replace("\r", "\n")

    paragraphs = re.split(r"\n\s*\n", text)

    cleaned_paragraphs = []

    for p in paragraphs:
        # Remove internal newlines and collapse whitespace
        single_line = " ".join(line.strip() for line in p.splitlines())
        single_line = re.sub(r"\s{2,}", " ", single_line).strip()

        if single_line:
            cleaned_paragraphs.append(single_line)

    return "\n\n".join(cleaned_paragraphs)


# ----------------------------
# Block segmentation + heuristics
# ----------------------------

def split_blocks(text: str, min_block_chars: int = 80) -> list[str]:
    """
    Paragraph-ish blocks separated by blank lines.
    """
    raw = [b.strip() for b in re.split(r"\n\s*\n", text) if b.strip()]
    return [b for b in raw if len(b) >= min_block_chars]


def validate_strip_param(value: int, name: str) -> int:
    if not (0 <= value <= 25):
        raise ValueError(f"{name} must be between 0 and 25 (inclusive), got {value}")
    return value


def strip_blocks(blocks: list[str], strip_pre: int = 0, strip_post: int = 0) -> list[str]:
    n = len(blocks)
    if n == 0:
        return blocks
    pre_n = int(n * (strip_pre / 100.0))
    post_n = int(n * (strip_post / 100.0))

    pre_n = min(pre_n, n)
    post_n = min(post_n, n - pre_n)

    return blocks[pre_n: n - post_n]


def is_toc_block(block: str, line_ratio: float = 0.4) -> bool:
    lines = [l for l in block.splitlines() if l.strip()]
    if len(lines) < 5:
        return False
    toc_like = sum(1 for l in lines if TOC_LINE_RE.match(l))
    return (toc_like / len(lines)) >= line_ratio


def is_index_block(block: str) -> bool:
    lines = [l.strip() for l in block.splitlines() if l.strip()]
    if len(lines) < 10:
        return False

    # mostly short lines
    if sum(len(l) < 60 for l in lines) / len(lines) < 0.7:
        return False

    # many lines end with numbers
    if sum(bool(re.search(r"\d+$", l)) for l in lines) / len(lines) < 0.6:
        return False

    # alphabetical diversity
    initials = {l[0].lower() for l in lines if l and l[0].isalpha()}
    return len(initials) >= 6


def split_reference_blocks(blocks: list[str]) -> tuple[list[str], list[str]]:
    main_blocks = []
    ref_blocks = []
    in_refs = False

    for b in blocks:
        first_line = (b.splitlines()[0].strip().lower() if b.splitlines() else "")
        if first_line in REFERENCE_HEADERS:
            in_refs = True

        if in_refs:
            ref_blocks.append(b)
        else:
            main_blocks.append(b)

    return main_blocks, ref_blocks


def normalize_blocks(blocks: list[str], min_chars: int = 220) -> list[str]:
    out = []
    for b in blocks:
        if len(b) < min_chars:
            continue
        # Skip screaming headings / garbage blocks
        if b.isupper():
            continue
        out.append(b)
    return out


# ----------------------------
# Main cleaning function
# ----------------------------

def clean_text_web_corpus(
    text: str,
    strip_pre: int = 0,
    strip_post: int = 0,
    cut_marker: str = CUT_MARKER
) -> dict:
    """
    Returns:
      {
        "clean_text": str,
        "references": Optional[str],
        "stats": dict
      }
    """
    original_chars = len(text)

    text = normalize_newlines(text)

    # 1) Cut at marker early (so removed tail doesn't affect heuristics)
    text = cut_after_marker(text, marker=cut_marker)

    # 2) Markdown + URL normalization
    text = replace_markdown_links(text)
    text = strip_bold_markers(text)
    text = strip_urls(text)

    text = remove_meta_lines(text)
    text = cleanup_whitespace(text)
    text = normalize_paragraph_linebreaks(text)

    # 3) Block-based structural filtering (page-agnostic)
    blocks = split_blocks(text)
    blocks_before_strip = len(blocks)

    blocks = strip_blocks(blocks, strip_pre=strip_pre, strip_post=strip_post)
    blocks_after_strip = len(blocks)

    # Remove TOC-like / index-like blocks
    blocks = [b for b in blocks if not is_toc_block(b) and not is_index_block(b)]

    # Separate references (excluded from clean_text by default)
    main_blocks, ref_blocks = split_reference_blocks(blocks)

    main_blocks = normalize_blocks(main_blocks)

    clean_text = "\n\n".join(main_blocks).strip() + "\n"
    references = ("\n\n".join(ref_blocks).strip() + "\n") if ref_blocks else None

    return {
        "clean_text": clean_text,
        "references": references,
        "stats": {
            "original_chars": original_chars,
            "clean_chars": len(clean_text),
            "blocks_before_strip": blocks_before_strip,
            "blocks_after_strip": blocks_after_strip,
            "strip_pre_pct": strip_pre,
            "strip_post_pct": strip_post,
            "cut_marker_found": (cut_marker in normalize_newlines(text)),  # note: post-cut marker is gone
        }
    }


# ----------------------------
# File IO + CLI
# ----------------------------

def should_ignore(path: Path) -> bool:
    return path.name.endswith("_cleaned.txt")


def cleaned_name(path: Path) -> str:
    # Keep extension .txt; append _cleaned to stem
    return f"{path.stem}_cleaned{path.suffix}"


def clean_file(
    input_path: Path,
    output_dir: Path,
    dry_run: bool,
    strip_pre: int,
    strip_post: int,
    cut_marker: str
) -> None:
    if should_ignore(input_path):
        print(f"[SKIP] Already cleaned: {input_path.name}")
        return

    with input_path.open("r", encoding="utf-8", errors="ignore") as f:
        raw = f.read()

    result = clean_text_web_corpus(
        raw,
        strip_pre=strip_pre,
        strip_post=strip_post,
        cut_marker=cut_marker
    )

    out_name = cleaned_name(input_path)
    out_path = output_dir / out_name

    orig_len = result["stats"]["original_chars"]
    clean_len = result["stats"]["clean_chars"]
    retained = (clean_len / orig_len * 100.0) if orig_len else 0.0

    if dry_run:
        print(
            f"[DRY-RUN] {input_path.name} → {out_name} | "
            f"retained: {retained:.1f}% ({clean_len}/{orig_len} chars) | "
            f"blocks: {result['stats']['blocks_before_strip']}→{result['stats']['blocks_after_strip']} "
            f"(strip_pre={strip_pre}%, strip_post={strip_post}%)"
        )
        return

    output_dir.mkdir(parents=True, exist_ok=True)

    if out_path.exists():
        # Don’t overwrite silently; this is corpus work.
        print(f"[SKIP] Output exists: {out_path}")
        return

    with out_path.open("w", encoding="utf-8") as f:
        f.write(result["clean_text"])

    print(f"[OK] {input_path.name} → {out_path}")


def clean_directory(
    input_dir: Path,
    output_dir: Path,
    dry_run: bool,
    strip_pre: int,
    strip_post: int,
    cut_marker: str
) -> None:
    if not input_dir.is_dir():
        raise ValueError(f"Input is not a directory: {input_dir}")

    files = sorted(input_dir.glob("*.txt"))
    if not files:
        print("[WARN] No .txt files found.")
        return

    for p in files:
        if should_ignore(p):
            print(f"[SKIP] Already cleaned: {p.name}")
            continue
        clean_file(
            p,
            output_dir=output_dir,
            dry_run=dry_run,
            strip_pre=strip_pre,
            strip_post=strip_post,
            cut_marker=cut_marker
        )


def main():
    parser = argparse.ArgumentParser(description="Clean a directory of .txt files into a target directory.")
    parser.add_argument("--input-dir", required=True, help="Directory containing .txt files to clean")
    parser.add_argument("--output-dir", required=True, help="Directory to write cleaned .txt files")
    parser.add_argument("--dry-run", action="store_true", help="Run cleaning without writing output files")
    parser.add_argument("--strip_pre", type=int, default=0, help="Percentage (0–25) of blocks to strip from the beginning")
    parser.add_argument("--strip_post", type=int, default=0, help="Percentage (0–25) of blocks to strip from the end")
    parser.add_argument(
        "--cut-marker",
        default=CUT_MARKER,
        help='Delete everything after this exact marker token (default: "**What you can do:**")'
    )

    args = parser.parse_args()

    input_dir = Path(args.input_dir).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()

    strip_pre = validate_strip_param(args.strip_pre, "--strip_pre")
    strip_post = validate_strip_param(args.strip_post, "--strip_post")

    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")

    clean_directory(
        input_dir=input_dir,
        output_dir=output_dir,
        dry_run=args.dry_run,
        strip_pre=strip_pre,
        strip_post=strip_post,
        cut_marker=args.cut_marker
    )


if __name__ == "__main__":
    main()