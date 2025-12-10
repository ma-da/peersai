#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -ne 2 ]; then
    echo "Usage: $0 <input_dir> <output_dir>"
    exit 1
fi 

INPUT_DIR="$1"
OUTPUT_DIR="$2"

# Check dirs
if [[ ! -d "$INPUT_DIR" ]]; then
    echo "Error: Input directory does not exist: $INPUT_DIR"
    exit 1
fi

mkdir -p "$OUTPUT_DIR"

# Log file for crashed PDFs
LOG_FILE="$OUTPUT_DIR/mineru_failures.log"
: > "$LOG_FILE"

echo "======================================================"
echo " MinerU Batch Conversion"
echo " Input:  $INPUT_DIR"
echo " Output: $OUTPUT_DIR"
echo "======================================================"
echo

# Find all PDFs recursively
mapfile -t pdfs < <(find "$INPUT_DIR" -type f \( -iname "*.pdf" \))

echo "Found ${#pdfs[@]} PDFs."
echo

# Process each PDF safely
for pdf in "${pdfs[@]}"; do
    rel_path="${pdf#$INPUT_DIR/}"
    base_name="$(basename "$pdf")"
    name="${base_name%.*}"

    # Build parallel output structure
    out_dir="$OUTPUT_DIR/$(dirname "$rel_path")/$name"
    mkdir -p "$out_dir"

    echo "Processing: $rel_path"
    echo "mineru -p $pdf -o $out_dir --method txt -b pipeline"

    if mineru \
        -p "$pdf" \
        -o "$out_dir" \
        --method txt \
        -b pipeline
    then
        echo "✓ Success: $rel_path"
    else
        echo "✗ Failed: $rel_path"
        echo "$rel_path" >> "$LOG_FILE"
    fi

    echo
done

echo "======================================================"
echo "All done!"
echo "Failures logged to: $LOG_FILE"
echo "Converted files in: $OUTPUT_DIR"
echo "======================================================"
