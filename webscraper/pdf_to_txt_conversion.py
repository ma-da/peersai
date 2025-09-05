#!/usr/bin/env python3

"""This file contains a tool for regenerating missing pdf files from the pdf corpus.
Optionally, if a '1' is passed as the first commandline argument, then will regenerate all txt.
"""
import os.path
from pathlib import Path

from bs4 import BeautifulSoup

import config
import sys

import content_filter
import pdf_fetcher
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
    #base_dir = Path(config.CORPUS_FOLDER_LOCATION)
    base_dir = Path("./")
    for pdf_file in base_dir.rglob("*.pdf"):
        pdf_filename = str(pdf_file)
        txt_filename = utils.get_txt_file_name(pdf_filename)

        # skip the txt file processing if we are in 'only_missing' mode and it exists
        if not regenerate_all and os.path.exists(txt_filename):
            print(f"txt file exists {txt_filename}")
            continue

        print(f"Processing pdf file {pdf_filename}", flush=config.FLUSH_LOG)
        with open(pdf_filename, 'rb') as input_file:
            try:
                print(f"Save PDF-to-text: {txt_filename}", flush=config.FLUSH_LOG)

                # Step 2: Extract text from the PDF
                #title, extracted_text = pdf_fetcher.extract_text_from_pdf(pdf_filename)
                title, extracted_text = pdf_fetcher.extract_clean_pdf_text(pdf_filename)

                # Step 3: Save the extracted text to a file
                pdf_fetcher.save_text_to_file(title, extracted_text, txt_filename)

                print(f"Regenerating pdf txt: '{txt_filename}' from file: '{pdf_file}'")
                n = n + 1
            except:
                print(f"Problem occurred when processing pdf file {pdf_filename}", flush=config.FLUSH_LOG)

    print(f"Text conversion converted {n} txt files")

if __name__ == "__main__":
    main()
