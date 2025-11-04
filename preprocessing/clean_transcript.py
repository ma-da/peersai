#!/usr/bin/env python
"""
clean_transcript.py
Remove speaker + timestamp lines from podcast/video transcripts.
Input:  raw transcript (one line per speaker or paragraph)
Output: clean dialogue only
"""

import argparse
import re
import sys
from pathlib import Path

# --------------------------------------------------------------
# 1. SPEAKER LINE DETECTION
# --------------------------------------------------------------
def is_speaker_line(line: str) -> bool:
    """
    Return True if line matches:
      • "Speaker X HH:MM:SS"
      • "Name HH:MM:SS"
      • "Name HH:MM:SS.sss"
    """
    line = line.strip()
    if not line:
        return False

    # Pattern:
    #   ^[A-Za-z]+(?:\s+[A-Za-z]+)*(?:\s+\d+)?\s+   → speaker name (1+ words, optional number)
    #   (?:\d{1,2}:\d{2}(?::\d{2})?(?:\.\d+)?)$     → time: MM:SS or HH:MM:SS + optional .sss
    pattern = r'^[A-Za-z]+(?:\s+[A-Za-z]+)*(?:\s+\d+)?\s+(?:\d{1,2}:\d{2}(?::\d{2})?(?:\.\d+)?)$'

    return bool(re.match(pattern, line))

# --------------------------------------------------------------
# 2. CLEANER
# --------------------------------------------------------------
def clean_transcript(input_path: Path, output_path: Path) -> None:
    """Read input, strip speaker lines, write clean output."""
    if not input_path.is_file():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    print(f"Cleaning: {input_path.name} → {output_path.name}")

    with input_path.open('r', encoding='utf-8', errors='ignore') as fin,\
         output_path.open('w', encoding='utf-8') as fout:

        cleaned_lines = []
        for line in fin:
            if is_speaker_line(line):
                continue  # skip speaker + timestamp
            cleaned_lines.append(line.rstrip())

        # Join with single newline, collapse empty lines
        clean_text = '\n'.join(cleaned_lines)
        clean_text = re.sub(r'\n{3,}', '\n\n', clean_text)  # max 1 blank
        fout.write(clean_text.strip() + '\n')

    print(f"Done! {len(cleaned_lines)} lines written.")

# --------------------------------------------------------------
# 3. CLI
# --------------------------------------------------------------
def main():

    parser = argparse.ArgumentParser(
        description="Clean podcast transcripts: remove speaker + timestamp lines"
    )
    parser.add_argument("input", help="Path to raw transcript file")
    parser.add_argument("output", help="Path to save cleaned transcript")
    args = parser.parse_args()

    try:
        clean_transcript(Path(args.input), Path(args.output))
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()