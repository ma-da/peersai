from config import *
import corpus_filtering

def main():
    files = [p for p in INPUT_DIR.rglob("*") if p.is_file() and p.suffix.lower() in EXTS]
    corpus_filtering.clean_corpus(files)

if __name__ == "__main__":
    main()