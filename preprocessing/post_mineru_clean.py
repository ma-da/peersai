import re
import argparse
from pathlib import Path

TOC_LINE_RE = re.compile(
    r'^\s*\d+(\.\d+)*\s+.*\.{3,}\s*\d+\s*$'
)


def validate_strip_param(value: int, name: str) -> int:
    if not (0 <= value <= 25):
        raise ValueError(
            f"{name} must be between 0 and 25 (inclusive), got {value}"
        )
    return value


def split_blocks(text: str, min_block_chars: int = 100) -> list[str]:
    """
    Split text into paragraph-like blocks.
    """
    raw_blocks = [b.strip() for b in text.split("\n\n") if b.strip()]
    return [b for b in raw_blocks if len(b) >= min_block_chars]


def strip_blocks(
    blocks: list[str],
    strip_pre: int = 0,
    strip_post: int = 0
) -> list[str]:
    """
    Strip a percentage of blocks from the start and/or end.
    strip_pre / strip_post are percentages (0–25).
    """
    n = len(blocks)
    if n == 0:
        return blocks

    pre_n = int(n * (strip_pre / 100.0))
    post_n = int(n * (strip_post / 100.0))

    # Clamp defensively
    pre_n = min(pre_n, n)
    post_n = min(post_n, n - pre_n)

    return blocks[pre_n : n - post_n]


def is_toc_block(block: str, line_ratio: float = 0.4) -> bool:
    lines = block.splitlines()
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

    # many numeric tails
    if sum(bool(re.search(r'\d+$', l)) for l in lines) / len(lines) < 0.6:
        return False

    # alphabetical diversity
    initials = {l[0].lower() for l in lines if l[0].isalpha()}
    return len(initials) >= 6

REFERENCE_HEADERS = {
    "references",
    "bibliography",
    "works cited",
    "literature cited"
}

def split_reference_blocks(blocks: list[str]) -> tuple[list[str], list[str]]:
    main_blocks = []
    ref_blocks = []

    in_refs = False
    for block in blocks:
        first_line = block.splitlines()[0].strip().lower()
        if first_line in REFERENCE_HEADERS:
            in_refs = True

        if in_refs:
            ref_blocks.append(block)
        else:
            main_blocks.append(block)

    return main_blocks, ref_blocks


def is_footnote_block(block: str) -> bool:
    lines = block.splitlines()
    if len(lines) > 5:
        return False

    return all(
        re.match(r'^\s*\(?\d+\)?\s+.+', l)
        for l in lines
    )


INLINE_FN_RE = re.compile(r'\[(\d+)\]|\((\d+)\)')

def inline_footnotes(text: str, footnotes: dict[str, str]) -> str:
    def repl(m):
        idx = m.group(1) or m.group(2)
        note = footnotes.get(idx)
        if note:
            return f" [FN: {note}]"
        return ""
    return INLINE_FN_RE.sub(repl, text)


def normalize_blocks(blocks: list[str], min_chars: int = 300) -> list[str]:
    cleaned = []
    for b in blocks:
        # skip heading-only blocks
        if len(b) < min_chars:
            continue
        # skip all-caps junk
        if b.isupper():
            continue
        cleaned.append(b)
    return cleaned


def clean_text_no_pages(
    text: str,
    strip_pre: int = 0,
    strip_post: int = 0
) -> dict:
    blocks = split_blocks(text)

    blocks = strip_blocks(
        blocks,
        strip_pre=strip_pre,
        strip_post=strip_post
    )

    blocks = [
        b for b in blocks
        if not is_toc_block(b)
        and not is_index_block(b)
    ]

    main_blocks, ref_blocks = split_reference_blocks(blocks)
    main_blocks = normalize_blocks(main_blocks)

    return {
        "clean_text": "\n\n".join(main_blocks),
        "references": "\n\n".join(ref_blocks) if ref_blocks else None
    }


def clean_single_file(
    input_path: Path,
    output_dir: Path,
    dry_run: bool = False,
    strip_pre: int = 0,
    strip_post: int = 0
) -> None:
    if input_path.name.endswith("_cleaned.txt"):
        print(f"[SKIP] Already cleaned: {input_path.name}")
        return

    output_name = input_path.stem + "_cleaned.txt"
    output_path = output_dir / output_name

    with input_path.open("r", encoding="utf-8", errors="ignore") as f:
        text = f.read()

    result = clean_text_no_pages(
        text,
        strip_pre=strip_pre,
        strip_post=strip_post
    )

    cleaned_text = result["clean_text"]

    orig_len = len(text)
    clean_len = len(cleaned_text)
    retained = (clean_len / orig_len * 100) if orig_len else 0.0

    if dry_run:
        print(
            f"[DRY-RUN] {input_path.name} → {output_name} | "
            f"retained: {retained:.1f}% "
            f"(strip_pre={strip_pre}%, strip_post={strip_post}%)"
        )
        return

    output_dir.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        f.write(cleaned_text)

    print(f"[OK] {input_path.name} → {output_path}")


def clean_directory(
    input_dir: Path,
    output_dir: Path,
    dry_run: bool = False,
    strip_pre: int = 0,
    strip_post: int = 0
) -> None:
    txt_files = sorted(input_dir.glob("*.txt"))

    for txt_file in txt_files:
        if txt_file.name.endswith("_cleaned.txt"):
            print(f"[SKIP] Already cleaned: {txt_file.name}")
            continue

        clean_single_file(
            txt_file,
            output_dir,
            dry_run=dry_run,
            strip_pre=strip_pre,
            strip_post=strip_post
        )


def main():
    parser = argparse.ArgumentParser(
        description="Clean TXT files produced from PDF/OCR pipelines"
    )

    parser.add_argument(
        "--input",
        required=True,
        help="Path to input .txt file OR directory containing .txt files"
    )

    parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory to write cleaned files"
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Run cleaning without writing output files"
    )

    parser.add_argument(
        "--strip_pre",
        type=int,
        default=0,
        help="Percentage (0–25) of blocks to strip from the beginning"
    )

    parser.add_argument(
        "--strip_post",
        type=int,
        default=0,
        help="Percentage (0–25) of blocks to strip from the end"
    )

    args = parser.parse_args()

    input_path = Path(args.input).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()

    strip_pre = validate_strip_param(args.strip_pre, "--strip_pre")
    strip_post = validate_strip_param(args.strip_post, "--strip_post")

    if not input_path.exists():
        raise FileNotFoundError(f"Input path not found: {input_path}")

    if input_path.is_file():
        clean_single_file(
            input_path,
            output_dir,
            dry_run=args.dry_run,
            strip_pre=args.strip_pre,
            strip_post=args.strip_post
        )
    elif input_path.is_dir():
        clean_directory(
            input_path,
            output_dir,
            dry_run=args.dry_run,
            strip_pre=args.strip_pre,
            strip_post=args.strip_post
        )
    else:
        raise ValueError("Input must be a file or directory")


if __name__ == "__main__":
    main()

