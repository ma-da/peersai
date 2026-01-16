#!/usr/bin/env bash

set -euo pipefail

SCRIPT_NAME="$(basename "$0")"

print_help() {
  cat <<EOF
Usage:
  $SCRIPT_NAME <source_dir> <target_dir>

Description:
  Updates .txt files in <target_dir> using files from <source_dir>.
  A file is copied only if:
    - it does not exist in the target directory, or
    - the source file is newer than the target file.

Options:
  -h, --help    Show this help message and exit

Safety:
  - Source and target directories are echoed before execution
  - Requires explicit confirmation (Y) before copying
  - Preserves file timestamps and permissions

Example:
  $SCRIPT_NAME ./src_txt ./dst_txt
EOF
}

# Help flag
if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  print_help
  exit 0
fi

# Argument count check
if [[ "$#" -ne 2 ]]; then
  echo "Error: Expected 2 arguments, got $#." >&2
  echo >&2
  print_help >&2
  exit 1
fi

SRC_DIR="$1"
DST_DIR="$2"

# Directory validation
if [[ ! -d "$SRC_DIR" ]]; then
  echo "Source directory does not exist: $SRC_DIR" >&2
  exit 1
fi

if [[ ! -d "$DST_DIR" ]]; then
  echo "Target directory does not exist: $DST_DIR" >&2
  exit 1
fi

echo "Source directory: $SRC_DIR"
echo "Target directory: $DST_DIR"
echo

read -r -p "Proceed with updating .txt files? [Y/n]: " CONFIRM

if [[ "$CONFIRM" != "Y" ]]; then
  echo "Aborted."
  exit 0
fi

echo

shopt -s nullglob

for src_file in "$SRC_DIR"/*.txt; do
  filename="$(basename "$src_file")"
  dst_file="$DST_DIR/$filename"

  if [[ ! -f "$dst_file" ]] || [[ "$src_file" -nt "$dst_file" ]]; then
    cp -p "$src_file" "$dst_file"
    echo "Copied: $filename"
  fi
done

echo
echo "Update complete."

