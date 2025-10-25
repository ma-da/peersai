# webscraper/cli.py
from __future__ import annotations
import argparse
import sys
from pathlib import Path
from multiprocessing import cpu_count, Pool
import re
from collections import Counter

import config as CFG
import cleaning_utils as cu

# --- Adapt calls to your current extractors ---
# HTML extractor example
import content_filter  # your existing module
# PDF extractor example
import pdf_fetcher     # your existing module
import epub_to_txt_conversion as epub_conv

def ensure_dir(p: Path | None):
    if p:
        p.mkdir(parents=True, exist_ok=True)

def process_single_html_file(src: Path, out_dir: Path, rej_dir: Path | None) -> tuple[Path, bool, str]:
    try:
        raw = src.read_bytes()
        decoded = cu.detect_and_decode(raw)
        title, body = content_filter.extract_content_newspaper(decoded)
        stripped = cu.strip_html_if_needed(body, force_html=True)
        clean = cu.normalize_and_clean(stripped)

        if not cu.has_natural_language_run(clean) or cu.is_garbled(clean):
            if rej_dir:
                (rej_dir / f"{src.stem}.reject.txt").write_text(clean, encoding="utf-8", newline="\n")
            return (src, False, "rejected")
        (out_dir / f"{src.stem}.txt").write_text(clean, encoding="utf-8", newline="\n")
        return (src, True, "ok")
    except Exception as e:
        if rej_dir:
            (rej_dir / f"{src.stem}.reject.txt").write_text("", encoding="utf-8", newline="\n")
        return (src, False, f"error: {e}")

def process_single_pdf_file(src: Path, out_dir: Path, rej_dir: Path | None) -> tuple[Path, bool, str]:
    try:
        title, extracted = pdf_fetcher.extract_clean_pdf_text(str(src))
        if isinstance(extracted, (bytes, bytearray)):
            extracted = cu.detect_and_decode(extracted)

        # get pages_count if available
        pages_count = None
        try:
            pages_count = pdf_fetcher.get_pages_count(str(src))
        except Exception:
            pass

        clean = clean_pdf_text(extracted, pages_count=pages_count)

        # accept/reject gate identical to your snippet
        if not cu.has_natural_language_run(clean) or cu.is_garbled(clean):
            if rej_dir:
                (rej_dir / f"{src.stem}.reject.txt").write_text(clean, encoding="utf-8", newline="\n")
            return (src, False, "rejected")

        (out_dir / f"{src.stem}.txt").write_text(clean, encoding="utf-8", newline="\n")
        return (src, True, "ok")

    except Exception as e:
        if rej_dir:
            (rej_dir / f"{src.stem}.reject.txt").write_text("", encoding="utf-8", newline="\n")
        return (src, False, f"error: {e}")

def process_single_epub_file(src: Path, out_dir: Path, rej_dir: Path | None, min_par_len: int) -> tuple[Path, bool, str]:
    #print(f"processing epub file {src}...")
    try:
        res = epub_conv.convert_epub_to_clean_text(src, min_paragraph_len=min_par_len)
        if res is None:
            print(f"failed to process epub file {src}...")
            if rej_dir:
                (rej_dir / f"{src.stem}.reject.txt").write_text("", encoding="utf-8", newline="\n")
            return (src, False, "rejected")
        title, clean = res
        (out_dir / f"{src.stem}.txt").write_text(clean, encoding="utf-8", newline="\n")
        #print(f"successfully processed epub file {src}.")
        return (src, True, "ok")
    except Exception as e:
        print(f"failed to process epub file {src}, exception {e}")
        if rej_dir:
            (rej_dir / f"{src.stem}.reject.txt").write_text("", encoding="utf-8", newline="\n")
        return (src, False, f"error: {e}")

def remove_repeated_headers_footers(text: str, min_occurrences: int = 0) -> str:
    """
    Remove very frequent short lines (likely headers/footers).
    Set min_occurrences > 0 to activate (e.g., pages_count * 0.6).
    """
    if min_occurrences <= 0:
        return text
    lines = [ln.strip() for ln in text.splitlines()]
    freq = Counter(ln for ln in lines if 2 <= len(ln) <= 80)
    to_drop = {ln for ln, n in freq.items() if n >= min_occurrences}
    if not to_drop:
        return text
    keep_lines = [ln for ln in lines if ln not in to_drop]
    out = "\n".join(keep_lines)
    out = re.sub(r"\n{3,}", "\n\n", out)
    return out

def clean_pdf_text(raw_text: str, pages_count: int | None = None) -> str:
    # Normalize/clean (includes de-hyphenation & whitespace fixes)
    txt = cu.final_clean_plain(raw_text)
    # Optional: drop repeated headers/footers if many pages
    if pages_count and pages_count >= 5:
        threshold = max(3, int(pages_count * 0.6))  # same heuristic as your code
        txt = remove_repeated_headers_footers(txt, min_occurrences=threshold)
        txt = cu.normalize_and_clean(txt)  # re-normalize after removal
    return txt      

def walk_inputs(root: Path, suffixes: set[str]) -> list[Path]:
    files: list[Path] = []
    for p in root.rglob("*"):
        if p.is_file() and p.suffix.lower() in suffixes:
            files.append(p)
    print(f"Found {len(files)} number of files to process")
    return files

def run_html(input_dir: Path, output_dir: Path, rejected_dir: Path | None, workers: int, dry_run: bool):
    ensure_dir(output_dir); ensure_dir(rejected_dir)
    items = walk_inputs(input_dir, {".html", ".htm", ".mhtml", ".xhtml", ".txt"})
    if dry_run:
        print(f"[DRY RUN] Would process {len(items)} HTML files from {input_dir} → {output_dir}")
        return
    if workers <= 1:
        results = [process_single_html_file(p, output_dir, rejected_dir) for p in items]
    else:
        with Pool(processes=workers) as pool:
            results = pool.starmap(process_single_html_file, [(p, output_dir, rejected_dir) for p in items])
    ok = sum(1 for _, s, _ in results if s)
    print(f"[HTML] {ok}/{len(items)} cleaned. Output: {output_dir}")

def run_pdf(input_dir: Path, output_dir: Path, rejected_dir: Path | None, workers: int, dry_run: bool):
    ensure_dir(output_dir); ensure_dir(rejected_dir)
    items = walk_inputs(input_dir, {".pdf"})
    if dry_run:
        print(f"[DRY RUN] Would process {len(items)} PDFs from {input_dir} → {output_dir}")
        return
    if workers <= 1:
        results = [process_single_pdf_file(p, output_dir, rejected_dir) for p in items]
    else:
        with Pool(processes=workers) as pool:
            results = pool.starmap(process_single_pdf_file, [(p, output_dir, rejected_dir) for p in items])
    ok = sum(1 for _, s, _ in results if s)
    print(f"[PDF] {ok}/{len(items)} cleaned. Output: {output_dir}")

def run_epub(input_dir: Path, output_dir: Path, rejected_dir: Path | None, workers: int, dry_run: bool, min_par_len: int):
    ensure_dir(output_dir); ensure_dir(rejected_dir)
    items = walk_inputs(input_dir, {".epub"})
    if dry_run:
        print(f"[DRY RUN] Would process {len(items)} EPUBs from {input_dir} → {output_dir}")
        return
    if workers <= 1:
        results = [process_single_epub_file(p, output_dir, rejected_dir, min_par_len) for p in items]
    else:
        from multiprocessing import Pool
        with Pool(processes=workers) as pool:
            results = pool.starmap(
                process_single_epub_file,
                [(p, output_dir, rejected_dir, min_par_len) for p in items]
            )
    ok = sum(1 for _, s, _ in results if s)
    print(f"[EPUB] {ok}/{len(items)} cleaned. Output: {output_dir}")

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="peersai-clean",
        description="Clean and convert crawled HTML/PDF/EPUB to training-ready TXT."
    )
    sub = p.add_subparsers(dest="cmd", required=True)

    # ---- helpers -------------------------------------------------------------
    def add_common(sp):
        sp.add_argument("--input-dir",  type=str, help="Override input directory")
        sp.add_argument("--output-dir", type=str, help="Override output directory")
        sp.add_argument("--rejected-dir", type=str, help="Where to save rejected outputs")
        sp.add_argument("--workers", type=int, default=cpu_count(),
                        help="Parallel workers (default: CPU count)")
        sp.add_argument("--dry-run", action="store_true",
                        help="List work without writing files")

    def add_pdf_opts(sp):
        sp.add_argument("--strip-headers", action="store_true",
                        help="Remove repeated headers/footers in PDFs")
        sp.add_argument("--header-min-frac", type=float, default=0.6,
                        help="Min fraction of pages a line must appear to be removed (PDF).")

    def add_epub_opts(sp):
        sp.add_argument("--min-paragraph-len", type=int, default=30,
                        help="Discard EPUB paragraphs shorter than this length.")

    # ---- html ---------------------------------------------------------------
    sp_html = sub.add_parser("html", help="Process HTML/TXT sources")
    add_common(sp_html)
    # (no format-specific flags)

    # ---- pdf ----------------------------------------------------------------
    sp_pdf = sub.add_parser("pdf", help="Process PDFs")
    add_common(sp_pdf)
    add_pdf_opts(sp_pdf)

    # ---- epub ---------------------------------------------------------------
    sp_epub = sub.add_parser("epub", help="Process EPUBs")
    add_common(sp_epub)
    add_epub_opts(sp_epub)

    # ---- all (html + pdf + epub) -------------------------------------------
    sp_all = sub.add_parser("all", help="Process HTML, then PDF, then EPUB")
    add_common(sp_all)
    add_pdf_opts(sp_all)
    add_epub_opts(sp_all)

    return p

def main(argv: list[str] | None = None):
    args = build_parser().parse_args(argv)

    # Resolve common paths (override config.py if flags are provided)
    out_dir = Path(args.output_dir or CFG.TEXT_OUTPUT_DIR).resolve()

    # rejected dir is optional
    rej_cfg = getattr(CFG, "REJECTED_OUTPUT_DIR", None)
    rej_src = args.rejected_dir if args.rejected_dir is not None else rej_cfg
    rej_dir = Path(rej_src).resolve() if rej_src else None

    workers = max(1, int(args.workers))
    dry_run = bool(args.dry_run)

    # Per-format input dirs (flags override config; for "all", the same --input-dir applies to all)
    html_in = Path(args.input_dir or getattr(CFG, "HTML_INPUT_DIR", "./html_raw")).resolve()
    pdf_in  = Path(args.input_dir or getattr(CFG, "PDF_INPUT_DIR", "./pdf_raw")).resolve()
    epub_in = Path(args.input_dir or getattr(CFG, "EPUB_INPUT_DIR", "./epub_raw")).resolve()

    # Format-specific knobs (present only on relevant subcommands)
    strip_headers   = getattr(args, "strip_headers", False)
    header_min_frac = float(getattr(args, "header_min_frac", 0.6))
    min_par_len     = int(getattr(args, "min_paragraph_len", 30))

    if args.cmd == "html":
        run_html(html_in, out_dir, rej_dir, workers, dry_run)

    elif args.cmd == "pdf":
        # Ensure your run_pdf signature accepts these two extra params
        run_pdf(pdf_in, out_dir, rej_dir, workers, dry_run)
        #run_pdf(pdf_in, out_dir, rej_dir, workers, dry_run,
        #        strip_headers=strip_headers, header_min_frac=header_min_frac)

    elif args.cmd == "epub":
        # Ensure your run_epub accepts min_par_len
        run_epub(epub_in, out_dir, rej_dir, workers, dry_run, min_par_len)

    elif args.cmd == "all":
        run_html(html_in, out_dir, rej_dir, workers, dry_run)
        run_pdf(pdf_in, out_dir, rej_dir, workers, dry_run)
        #run_pdf(pdf_in,  out_dir, rej_dir, workers, dry_run,
        #        strip_headers=strip_headers, header_min_frac=header_min_frac)
        run_epub(epub_in, out_dir, rej_dir, workers, dry_run, min_par_len)

    else:
        raise SystemExit("Unknown command")


if __name__ == "__main__":
    main()

