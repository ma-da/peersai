#!/bin/bash
mkdir -p ./txt_corpus
echo "Copied corpus txt files to txt_corpus/"
cp ./corpus/*.txt ./txt_corpus/
echo "Zipping to ./txt_corpus.tar.gz"
tar czvf ./txt_corpus.tar.gz ./txt_corpus
