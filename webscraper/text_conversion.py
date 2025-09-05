#!/usr/bin/env python3

"""This file contains a tool for regenerating missing text files from the html corpus.
Optionally, if a '1' is passed as the first commandline argument, then will regenerate all.
"""
import os.path
from pathlib import Path

from bs4 import BeautifulSoup

import config
import sys

import content_filter
import utils


def main():
    regenerate_all = False
    if len(sys.argv) > 1:
        if len(sys.argv[1]) > 0 and sys.argv[1] == '1':
            print("Setting txt conversion mode to <all>")
            regenerate_all = True
    else:
        print("Setting txt conversion mode to <only_missing>")

    n = 0
    base_dir = Path(config.CORPUS_FOLDER_LOCATION)
    for html_file in base_dir.rglob("*.html"):
        html_filename = str(html_file)
        txt_filename = utils.get_txt_file_name(html_filename)

        # skip the txt file processing if we are in 'only_missing' mode and it exists
        if not regenerate_all and os.path.exists(txt_filename):
            continue

        print(f"Processing html file {html_filename}", flush=config.FLUSH_LOG)
        with open(html_filename, 'rb') as input_file:
            try:
                content = input_file.read()
                soup = BeautifulSoup(content, 'html.parser')

                if not content or len(content) == 0:
                    continue

                # here we want to choose the appropriate content conversion strategy
                title, txt_content = content_filter.extract_content_newspaper(content)
                #txt_content = content_filter.extract_content_from_soup(soup)

                with open(txt_filename, 'wb') as output_file:
                    output_file.write((title + "\n\n").encode("utf-8"))
                    output_file.write(txt_content.encode("utf-8"))
                    print(f"Regenerating txt: '{txt_filename}' from file: '{html_file}'")
                    n = n + 1
            except:
                print(f"Problem occurred when processing html file {html_filename}", flush=config.FLUSH_LOG)

    print(f"Text conversion converted {n} txt files")

if __name__ == "__main__":
    main()
