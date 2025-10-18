"""
The script takes an input directory of epub files and converts them to txt in an output directory.

Deps:
    pip install ebooklib beautifulsoup4
"""

import os
import re
import sys
from pathlib import Path
from ebooklib import epub
from bs4 import BeautifulSoup

def convert_epub_to_txt(epub_path: Path, output_dir: Path, min_paragraph_len: int = 30) -> Path:
    """
    Convert a single EPUB file to a cleaned TXT file.

    Parameters
    ----------
    epub_path : Path
        Path to the EPUB file.
    output_dir : Path
        Directory to write the TXT file.
    min_paragraph_len : int
        Minimum paragraph length (shorter lines are discarded).
    """

    book = epub.read_epub(str(epub_path))
    all_texts = []

    for item in book.get_items():
        if item.get_type() == epub.ITEM_DOCUMENT:
            soup = BeautifulSoup(item.get_body_content(), "html.parser")

            # Remove non-textual elements
            for tag in soup(["script", "style", "header", "footer", "nav", "aside"]):
                tag.decompose()

            # Extract visible text
            text = soup.get_text(separator="\n", strip=True)
            text = re.sub(r"\n{2,}", "\n\n", text)
            paragraphs = [p.strip() for p in text.split("\n") if len(p.strip()) >= min_paragraph_len]
            all_texts.extend(paragraphs)

    full_text = "\n\n".join(all_texts)

    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / (epub_path.stem + ".txt")

    with output_path.open("w", encoding="utf-8") as f:
        f.write(full_text)

    print(f"✅ {epub_path.name} → {output_path.name} ({len(all_texts)} paragraphs)")
    return output_path


def batch_convert_epubs(input_dir: str, output_dir: str, min_paragraph_len: int = 30):
    """
    Convert all EPUB files in input_dir to TXT files in output_dir.
    """
    in_dir = Path(input_dir)
    out_dir = Path(output_dir)

    epub_files = list(in_dir.rglob("*.epub"))
    if not epub_files:
        print(f"No EPUB files found in {in_dir}")
        return

    print(f"Found {len(epub_files)} EPUB file(s) in {in_dir}")
    for epub_path in epub_files:
        try:
            convert_epub_to_txt(epub_path, out_dir, min_paragraph_len)
        except Exception as e:
            print(f"❌ Failed to process {epub_path.name}: {e}")


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python batch_epub_to_txt.py <input_dir> <output_dir>")
        sys.exit(1)

    input_dir = sys.argv[1]
    output_dir = sys.argv[2]

    batch_convert_epubs(input_dir, output_dir)
