#!/bin/bash

if [ "$#" -ne 1 ]; then
        echo "Please input target dir"
        exit 1
fi

src_dir=$1

# add slash if missing
[[ "${src_dir}" != */ ]] && path="${src_dir}/"

# Batch convert all .md → .txt (recursive)
find $1 -name "*.md" | while read md; do
    txt="${md%.md}.txt"
    echo "Creating target file ${txt}..."
    pandoc "$md" -t plain --wrap=none -o "$txt"
done
