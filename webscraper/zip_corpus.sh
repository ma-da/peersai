#!/bin/bash
mkdir -p ./txt_corpus

# Format: YYYY-MM-DD_HH-MM-SS
timestamp=$(date "+%Y-%m-%d_%H-%M-%S")

# Output filename
tar_filename="./corpus_archive_${timestamp}.tar.gz"

echo "Zipping corpus..."
tar -czf "$tar_filename" ./corpus
