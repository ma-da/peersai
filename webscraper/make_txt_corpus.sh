#!/bin/bash
echo "Zipping all .txt files in corpus/ to ./txt_corpus.tar.gz"
find ./corpus -type f -name "*.txt" -print0 | tar --null -czf ./txt_corpus.tar.gz --files-from=-

