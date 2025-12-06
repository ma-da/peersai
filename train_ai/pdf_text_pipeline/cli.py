# Simple CLI to run extraction -> normalization
import argparse
from src.pdf_pipeline.extract import hybrid_extract
from src.pdf_pipeline.normalize import normalize_intermediate_json
from src.pdf_pipeline.evaluate import compare_files
import os
import json

def main():
    ap = argparse.ArgumentParser(description='PDF to clean JSON extraction pipeline')
    ap.add_argument('pdf', help='Path to PDF file')
    ap.add_argument('--out-dir', default='output', help='Directory to write outputs')
    ap.add_argument('--pdftotext', default='pdftotext', help='pdftotext binary name if available')
    ap.add_argument('--ocr-lang', default=None, help='pytesseract language (optional)')
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    intermediate = os.path.join(args.out_dir, 'intermediate.json')
    cleaned = os.path.join(args.out_dir, 'cleaned.json')
    print('Running hybrid extraction...')
    hybrid_extract(args.pdf, intermediate, pdftotext_path=args.pdftotext, ocr_lang=args.ocr_lang)
    print('Running normalization...')
    normalize_intermediate_json(intermediate, cleaned)
    print('Done. Outputs:')
    print(' - intermediate:', intermediate)
    print(' - cleaned:', cleaned)
    try:
        # if another baseline exists, evaluate
        baseline = os.path.join(args.out_dir, 'baseline.json')
        if os.path.exists(baseline):
            print('Comparing with baseline.json...')
            summary = compare_files(baseline, cleaned)
            print(json.dumps(summary, indent=2))
    except Exception as e:
        print('Evaluation error:', e)

if __name__ == '__main__':
    main()
