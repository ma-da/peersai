# peers ai webscraper
The webscraper for downloading peers-family corpus training material

See config settings in config.py before running.
- you will likely want to change MAX_PAGES_CRAWL_LIMIT (set to 0 for no limit)

To run the web crawler:
./run.sh [optional max number of pages to crawl, omit to crawl up to config limit]

To only regenerate missing txt files from corpus files:
./regen_missing_txt.sh

To regenerate all missing txt files from corpus files:
./regen_all_txt.sh

## cli.py text cleaning utility

Directory paths are defined in webscraper/config.py. Override any of these with command-line flags (--input-dir, --output-dir, --rejected-dir).

Example entries
HTML_INPUT_DIR = "./corpus/html_raw"
PDF_INPUT_DIR = "./corpus/pdf_raw"
TEXT_OUTPUT_DIR = "./corpus/clean_text"
REJECTED_OUTPUT_DIR = "./corpus/rejected"

1. python -m venv .venv
-Windows
.venv\Scripts\activate
-macOS/Linux
source .venv/bin/activate

2. From the repository root (peersai/):
pip install -U pip
pip install -e .
This registers the console script peersai-clean globally in your venv.
3. Run commands:

peersai-clean --help
Options:
--input-dir PATH        Override input directory
--output-dir PATH       Override output directory
--rejected-dir PATH     Directory to save rejected files
--workers N             Number of parallel workers (default = CPU count)
--dry-run               Preview actions without writing output
--strip-headers         Remove repeated headers/footers (PDF only)
--header-min-frac F     Minimum fraction of pages a line must appear to be removed
Clean using defaults from config.py
peersai-clean html
peersai-clean pdf
peersai-clean all

