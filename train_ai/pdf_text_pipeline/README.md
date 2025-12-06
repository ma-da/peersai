# PDF Text Extraction & Normalization Pipeline

This package provides a practical, modular pipeline for extracting clean, structured text from PDFs suitable for ML training.

Features:
- Hybrid extraction that chooses between text-layer extraction and OCR per-document
- Intermediate JSON with per-page metadata
- Unicode normalization, header/footer heuristics, and intelligent line-joining
- Simple CLI and evaluation tool to compare two outputs

Installation (recommended inside virtualenv or Docker):
1. Install system deps:
   - Poppler (for pdftotext and pdf2image): e.g. `sudo apt-get install poppler-utils poppler-data`
   - Tesseract OCR: e.g. `sudo apt-get install tesseract-ocr`
2. Install Python deps:
   ```bash
   pip install -r requirements.txt
   ```

Example usage:
```bash
python cli.py sample.pdf --out-dir results
```

Outputs:
- `results/intermediate.json` -> raw extraction with source and confidence
- `results/cleaned.json` -> normalized, ready-for-training text per page

Notes & tradeoffs:
- The pipeline favors conservative heuristics. For large corpora, use the evaluate.compare_files function to detect pathological PDFs.
- Keep the intermediate JSON; it contains metadata useful for re-joining or re-processing without re-running expensive OCR.
