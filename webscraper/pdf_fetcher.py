#!/usr/bin/env python3

"""This module provides utility functions related to PDF retrieval and text extraction.
"""

import requests
import fitz  # PyMuPDF
import config

def download_pdf(url, output_path):
    """Download a PDF file from a URL and save it locally."""
    response = requests.get(url)
    if response.status_code == 200:
        with open(output_path, 'wb') as file:
            file.write(response.content)
        print(f"PDF downloaded and saved to {output_path}")
    else:
        print(f"Failed to download PDF. Status code: {response.status_code}")


def extract_text_from_pdf(pdf_path):
    """Extract text from a PDF file using PyMuPDF."""
    text = ""
    try:
        # Open the PDF file
        with fitz.open(pdf_path) as doc:
            # Iterate through each page
            for page in doc:
                # Extract text from the page
                text += page.get_text()
        print("Text extracted successfully.")
    except Exception as e:
        print(f"Error extracting text: {e}")
    return text


def save_text_to_file(text, output_path):
    """Save extracted text to a file."""
    with open(output_path, 'w', encoding='utf-8') as file:
        file.write(text)
    print(f"Text saved to {output_path}")


# Example usage
if __name__ == "__main__":
    # URL of the PDF file
    pdf_url = config.TEST_PDF_URL

    # Local paths
    pdf_output_path = config.CORPUS_FOLDER_LOCATION + "sample.pdf"  # Path to save the downloaded PDF
    text_output_path = config.CORPUS_FOLDER_LOCATION + "output.txt"  # Path to save the extracted text

    # Step 1: Download the PDF
    download_pdf(pdf_url, pdf_output_path)

    # Step 2: Extract text from the PDF
    extracted_text = extract_text_from_pdf(pdf_output_path)

    # Step 3: Save the extracted text to a file
    save_text_to_file(extracted_text, text_output_path)

    # Print the extracted text (optional)
    print("Extracted Text:\n", extracted_text)
