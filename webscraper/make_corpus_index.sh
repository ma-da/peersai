#!/bin/bash
if [ "$#" -ne 1 ]; then
    echo "Please select corpus directory"
    exit 1
fi

echo "Making corpus list for corpus in directory: $1"

target=./corpus_file_list.txt
ls -ltr $1 | awk '{print $9, $5}' | sort -k 1 -k 2n > $target
echo "Created corpus file list in $target"

cat ./corpus_file_list.txt | sort > ./corpus_index.txt
echo "Created corpus file list in ./corpus_index.txt"
