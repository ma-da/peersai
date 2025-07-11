#!/usr/bin/env python3

# config.py
# contains various project settings

# Set your corpus location
CORPUS_FOLDER_LOCATION = "./corpus/"                           # Nate's setting
#CORPUS_FOLDER_LOCATION = "C:\\Users\\rames\\ai\\CrawlTest\\"  # Marc's setting

# Set your log file locations
LOGS_FOLDER_LOCATION = "./logs/"                            # Nate's setting
#LOGS_FOLDER_LOCATION = "C:\\Users\\rames\\ai\\CrawlTest\\"  # Marc's setting

LOGS_NAME = "scraper.log"

# language setting for python function recursion depth
PYTHON_RECURSION_DEPTH = 2000

# stop crawling after this number of pages (set to 0 to disable)
MAX_PAGES_CRAWL_LIMIT = 10

# how far done to crawl non-peers pages
# Deprecated - we are not crawling links for non-peers pages
MAX_DEPTH_CRAWL_LIMIT = 2

# crawler behavior
ENABLE_PROCESS_PDFS = False  # for testing
SAVE_HTML_CONTENT = True    # for testing
FLUSH_LOG = False            # set to true for better debugging output
DEBUG_ENABLED = True         # set to False to disable debug printing

# test url for testing pdf functionality
TEST_PDF_URL = "https://ontheline.trincoll.edu/images/bookdown/sample-local-pdf.pdf"

# Wayback Machine API endpoint
WAYBACK_API = "http://archive.org/wayback/available"
headers = {"User-Agent": "AiBot/1.0"}
