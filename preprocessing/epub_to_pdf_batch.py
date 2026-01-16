#!/usr/bin/env python3

import argparse
import subprocess
import sys
from pathlib import Path

"""
Batch conversion of epub files to pdf

You will need calibre to run:
    # macOS
    brew install --cask calibre

    # Ubuntu / Debian
    sudo apt install calibre

Verify you have calibre installed using:
    ebook-convert --version

How to run:
    python epub_to_pdf_batch.py ./epubs ./pdfs
"""

def convert_epub_to_pdf(src_dir: Path, dst_dir: Path) -> None:
    if not src_dir.exists() or not src_dir.is_dir():
        raise ValueError(f"Source directory does not exist: {src_dir}")

    dst_dir.mkdir(parents=True, exist_ok=True)

    epub_files = list(src_dir.glob("*.epub"))

    if not epub_files:
        print("No EPUB files found.")
        return

    for epub_path in epub_files:
        pdf_path = dst_dir / (epub_path.stem + ".pdf")

        print(f"Converting: {epub_path.name} → {pdf_path.name}")

        try:
            subprocess.run(
                [
                    "ebook-convert",
                    str(epub_path),
                    str(pdf_path),
                    "--paper-size", "letter",
                    "--margin-left", "36",
                    "--margin-right", "36",
                    "--margin-top", "36",
                    "--margin-bottom", "36",
                ],
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
        except subprocess.CalledProcessError as e:
            print(f"❌ Failed to convert {epub_path.name}")
            print(e.stderr.decode(errors="ignore"))
        else:
            print(f"✅ Done: {pdf_path.name}")


def main():
    parser = argparse.ArgumentParser(
        description="Batch convert EPUB files to PDF"
    )
    parser.add_argument(
        "src",
        type=Path,
        help="Source directory containing EPUB files",
    )
    parser.add_argument(
        "dst",
        type=Path,
        help="Target directory for generated PDFs",
    )

    args = parser.parse_args()

    try:
        convert_epub_to_pdf(args.src, args.dst)
    except Exception as e:
        print(f"Fatal error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
