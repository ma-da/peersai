#!/usr/bin/env python3

"""This file contains a tool for converting a single html file to a txt file
Writes to txt file name and also single_output.txt in the corpus
"""
import os.path
from pathlib import Path

from bs4 import BeautifulSoup

import config
import sys

import content_filter
import utils


def main():
    if len(sys.argv) < 2:
        print("Please pass an html input file")
        return

    input_filename = sys.argv[1]
    print(f"Processing converting of html file {input_filename}")

    if not utils.is_html_file(input_filename):
        print(f"Not an html file.")
        return

    with open(input_filename, 'rb') as input_file:
        content = input_file.read()
        soup = BeautifulSoup(content, 'html.parser')

        #txt_content = content_filter.extract_content_from_soup(soup)
        txt_content = content_filter.extract_content_newspaper(content)
        if txt_content is None:
            print("Problem occurred when filtering content")
            return

        txt_filename = input_filename.replace('.html', '.txt')
        print(f"Writing output txt file: {txt_filename}")
        with open(txt_filename, 'wb') as output_file:
            output_file.write(txt_content.encode("utf-8"))
            print(f"Regenerating txt: '{txt_filename}'")

        output_filename = config.CORPUS_FOLDER_LOCATION + "/output.txt"
        print(f"Writing output log file: {output_filename}")
        with open(output_filename, 'wb') as output_file:
            output_file.write(txt_content.encode("utf-8"))

if __name__ == "__main__":
    main()
