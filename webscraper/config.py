#!/usr/bin/env python3
import re

# config.py
# contains various project settings

# Set your corpus location
CORPUS_FOLDER_LOCATION = "./corpus/"                           # Nate's setting
#CORPUS_FOLDER_LOCATION = "C:\\Users\\rames\\ai\\CrawlTest\\"  # Marc's setting

# Set your db cache metadata location
DB_CACHE_LOCATION = "./db_cache/"
DB_CACHE_NAME = "meta_cache.db"
DB_CACHE_PATH = DB_CACHE_LOCATION + DB_CACHE_NAME

# Set your log file locations
LOGS_FOLDER_LOCATION = "./logs/"                            # Nate's setting
#LOGS_FOLDER_LOCATION = "C:\\Users\\rames\\ai\\CrawlTest\\"  # Marc's setting

# log file for scraper operations
LOGS_NAME = "scraper.log"

# language setting for python function recursion depth
PYTHON_RECURSION_DEPTH = 2000

# stop crawling after this number of pages (set to 0 to disable)
MAX_PAGES_CRAWL_LIMIT = 0

# how far done to crawl non-peers pages
# Deprecated - we are not crawling links for non-peers pages
MAX_DEPTH_CRAWL_LIMIT = 2

# progress output every N pages
# this number cannot be zero
PROGRESS_REPORT_N_PAGES = 25

# settings used if we encounter a ratelimit response (http code 429)
RATELIMIT_RETRIES = 3
RATELIMIT_RETRY_TIME_SECS = 0.1

# crawler behavior
ENABLE_PROCESS_PDFS = True          # for testing
SAVE_HTML_CONTENT = True            # for testing
FLUSH_LOG = True                   # set to true for better debugging output
DEBUG_ENABLED = True                # set to False to disable debug printing
CACHE_ENABLED = True               # set to true to enable cache usage
FLUSH_CACHE_ON_START = False        # set to true to flush the cache on program start
LOAD_PENDING_QUEUE_ON_START = True  # set to true to load the pending queue on start (pending prioritized ahead of start_url)

# enumerates possible crawler fetch strategy
CRAWLER_FETCH_REQUESTS = 1
CRAWLER_FETCH_PLAYWRIGHT = 2

# this is the fetch strategy that the crawler will use
CRAWLER_FETCH_STRATEGY = CRAWLER_FETCH_REQUESTS
#CRAWLER_FETCH_STRATEGY = CRAWLER_FETCH_PLAYWRIGHT

# test url for testing pdf functionality
TEST_PDF_URL = "https://ontheline.trincoll.edu/images/bookdown/sample-local-pdf.pdf"

# Wayback Machine API endpoint
WAYBACK_API = "http://archive.org/wayback/available"
headers = {"User-Agent": "AiBot/1.0"}

# These are the sites we will be visiting
pattern_peers_family = re.compile(r"""
    ^https?://                          # Start with http or https
    (www\.)?                            # Literal 'www.'
    (                                   # Start of group for domain names
        wanttoknow\.info         |
        momentoflove\.org        |
        weboflove\.org           |
        newsarticles\.media      |
        divinemystery\.net       |
        inspiringcommunity\.org  |
        wisdomcourses\.net       |
        inspirationcourse\.net   |
        hidden-knowledge\.net    |
        insightcourse\.net       |
        transformationteam\.net  |
        martintruther.com        |
        gatheringspot\.net       |
        usrtk\.org                |
        newparadigminstitute\.org |
        responsibletechnology\.org |
        theblackvault\.com         |
        farsight\.org              |
        childrenshealthdefense\.org
    )                                   # End of group
    (?=/|$)                             # Ensure domain ends properly (optional but safer)
    """, re.VERBOSE | re.IGNORECASE)

pattern_archive_url = re.compile(r"""
    ^https?://        # Start with http or https
    web\.archive\.org
    """, re.VERBOSE | re.IGNORECASE)

pattern_hash_url = r'#([\w-]+)$'

# don't visit any urls that match this
pattern_filter_list = re.compile(r"""
    ^java?script:              |  # javascript links
    ^mailto:                   |  # mailto links
    amazon\.com                |
    youtube\.com               |
    youtu\.be                  |
    instagram\.com             |
    facebook\.com              |
    tiktok\.com                |
    twitter\.com               |
    x\.com                     |  # Twitter's new domain
    linkedin\.com              |
    reddit\.com                |
    pinterest\.com             |
    snapchat\.com              |
    nytimes\.com               |  # New York Times
    washingtontimes\.com       |  # Washington Times
    cnn\.com                   |
    foxnews\.com               |
    nbcnews\.com               |
    abcnews\.go\.com           |
    example\.com               |
    example\.org               |
    rumble\.com                |
    redirect                   |
    sign-in                    |
    wa\.me                     |
    tel:\+                     |
    \.gov\b                    |  # any domain ending with .gov
    \.mil\b                       # military domains
""", re.VERBOSE | re.IGNORECASE)

