#!/usr/bin/env python3

# TITLE: PEERS AI Project Website Scraper (WantToKnowScraper)
# VERSION: 1.0
# AUTHOR: Marc Baber
# DATE: 06 MAR 2025
#
# TO DO LIST:
# 1. Make FIFO stack, for breadth first, not depth
# 2. No hash suffixes
# 3. Don't take HREFs from archived docs
# 4. Need a way to detect and avoid recursion. PRUNE???
# 5. If file already local, load it instead???
# 6. Prune out MK docs
#
#
# 1. Fix MomentOfLove navbar recursions on g/victim_or_creator_vs and inspiration/inspiring-videos
# 2. Set recursion limit higher (apparently 1000 wasn't high enough). But is there a way to avoid getting so deep?
#    I might have to go breadth first (instead of depth first) using a FIFO set/stack
#    a. seed the FIFO with just the first (base) URL
#    b. WHILE there's anything in the FIFO queue that has not already been visited:
#       crawl the first URL:
#       retrieve file (html or pdf) -- getting archive if necessary
#       make .txt file
#       If HTML, add all hrefs to FIFO stack, but not if archive (push right)
#       Add this url to visited.
#       WEND
# 3. Don't visit a # hash href if the base URL has already been visited
# 4. Why did processing stop after 50 hrs on a PPT file: http://www.cs.cmu.edu/~pausch/Randy/Randy/pauschlastlecture.ppt ?
#    (update: did not crash, just stalled)
# 5.


import config
from pybloom_live import BloomFilter
import pdf_fetcher
import os
import requests
import re
import fitz  # PyMuPDF
import sys
from utils import *
from bs4 import BeautifulSoup
from urllib.parse import urljoin

# Don't prune page tree while still on "home site" which should include whole family of PEERS
# Websites and online courses currently managed by PEERS:
#
# www.momentoflove.org - Every person in the world has a heart
# www.weboflove.org - Strengthening the web of love that interconnects us all
# www.WantToKnow.info - Revealing major cover-ups and working together for a brighter future
# www.newsarticles.media - Collection of under-reported major media news articles
# www.divinemystery.net - Mystical musings of a spiritual explorer
# www.inspiringcommunity.org - Building a global community for all
# www.wisdomcourses.net - Free online courses inspire you to greatness
# www.inspirationcourse.net - The Inspiration Course: Opening to more love and deeper connection
# www.hidden-knowledge.net - Hidden Knowledge Course: Illuminating shadow aspects of our world
# www.insightcourse.net - The Insight Course: The best of the Internet all in one free course
# www.transformationteam.net - Transformation Team: Building bridges to expanded consciousness
# www.gatheringspot.net - Dynamic community networking portal for course graduates

pattern_peers_family = re.compile(r"""
    ^https?://        # Start with http or https
    www\.             # Literal 'www.'
    (                 # Start of group for domain names
        wanttoknow\.info         |
        momentoflove\.org        | 
        # momentoflove domain taken out 03/06/25 due to recursive href's in the navbar in two places:
        # 1. g/victim_or_creator_vs lead to "g/g/g/g/g/g/g/g" recursions
        # 2. inspiration/inspiring-videos lead to "inspiration/inspiration/inspiration..." recursions
        weboflove\.org           |
        newsarticles\.media      |
        divinemystery\.net       |
        inspiringcommunity\.org  |
        wisdomcourses\.net       |
        inspirationcourse\.net   |
        hidden-knowledge\.net    |
        insightcourse\.net       |
        transformationteam\.net  |
        gatheringspot\.net
    )                # End of group
    """, re.VERBOSE | re.IGNORECASE)

pattern_archive_url = re.compile(r"""
    ^https?://        # Start with http or https
    web\.archive\.org        
    """, re.VERBOSE | re.IGNORECASE)

pattern_hash_url = r'#([\w-]+)$'

# don't visit any urls that match this
pattern_filter_list = re.compile(r"""
    javscript:;
""", re.VERBOSE | re.IGNORECASE
                                 )


# Check if this url should be processed
# Note preferable check this prior to recursive call of crawl
def should_visit(url, depth_effective, visited_set):
    debug(f"Should visit url {url} depth {depth_effective}?", flush=config.FLUSH_LOG)

    if pattern_filter_list.match(url):
        debug(f"Visit declined. Skipping pattern filter list {url}", flush=config.FLUSH_LOG)
        return False

    if pattern_archive_url.match(url):
        debug(f"Visit declined. Skipping Archive URL {url}", flush=config.FLUSH_LOG)
        return False

    if url in visited_set:
        debug(f"Visit declined. Previously visited: {url}", flush=config.FLUSH_LOG)
        return False

    # Don't process image files
    if (bool(re.search('.jpe+g$', url)) or bool(re.search('.gif$', url)) or bool(re.search('.png$', url))):
        debug(f"Visit declined. Skipping image: {url}", flush=config.FLUSH_LOG)
        return False

    # Don't process mailto's
    if (bool(re.search('^mailto:', url))):
        debug(f"Visit declined. Skipping mailto: {url}", flush=config.FLUSH_LOG)
        return False

    debug(f"Accepted visit url {url} depth {depth_effective}", flush=config.FLUSH_LOG)
    return True


def should_process_child_links(depth_effective, is_peers_family, max_depth):
    # don't stray too far from home domain(s)
    if depth_effective >= max_depth:
        debug(f"process_child_links declined. Effective depth exceeded {depth_effective}", flush=config.FLUSH_LOG)
        return False

    # only visit peers family site
    if not is_peers_family:
        debug(f"process_child_links declined. Not a peers family site", flush=config.FLUSH_LOG)
        return False

    debug(f"process child links allowed", flush=config.FLUSH_LOG)
    return True


def crawl_site(start_url, output_dir, max_depth=2, max_pages=-1):
    os.makedirs(output_dir, exist_ok=True)
    visited = set()
    seen_content_hashes = BloomFilter(capacity=1_000_000, error_rate=0.00001)
    num_pages_visited = 0

    # crawl should process only already filtered urls
    def crawl(url, depth_actual, depth_effective):
        nonlocal num_pages_visited

        is_peers_family = False
        if pattern_peers_family.match(url):
            depth_effective = 0  # Effective depth is how many hops from home domain(s)
            is_peers_family = True
            debug(f"URL is in Home Domain(s): {url}", flush=config.FLUSH_LOG)
        else:
            debug(f"URL is NOT in Home Domain(s) {url}", flush=config.FLUSH_LOG)

        num_pages_visited = num_pages_visited + 1
        if max_pages > 0 and num_pages_visited > max_pages:
            print("Maximum pages hit. Stopping crawl.")
            raise StopIteration("Maximum pages hit")

        debug(f"Adding {url} to visited set")
        visited.add(url)

        print(f"({depth_actual}/{depth_effective}) CRAWLING: {url}", flush=config.FLUSH_LOG)

        # Try to fetch the page
        try:
            response = requests.get(url, headers=config.headers, timeout=15)
            if response.status_code == 200:
                clean_url = url.replace('https://', '')
                clean_url = clean_url.replace('http://', '')
                clean_url = clean_url.rstrip('/')

                content_type = response.headers.get('Content-Type')

                if config.ENABLE_PROCESS_PDFS and 'application/pdf' in content_type:
                    print(f"File appears to be PDF {url}", flush=config.FLUSH_LOG)

                    pdf_output_path = os.path.join(output_dir, clean_url.replace('/', '_') + '.pdf')
                    pdf_output_txt_filename = pdf_output_path.replace('.pdf', '.txt')
                    print(f"Save PDF-to-text: {pdf_output_txt_filename}", flush=config.FLUSH_LOG)

                    # Step 1: Download the PDF
                    pdf_fetcher.download_pdf(url, pdf_output_path)

                    # Step 2: Extract text from the PDF
                    extracted_text = pdf_fetcher.extract_text_from_pdf(pdf_output_path)

                    # Step 3: Save the extracted text to a file
                    pdf_fetcher.save_text_to_file(extracted_text, pdf_output_txt_filename)

                elif 'text/html' in content_type:
                    debug(f"File appears to be HTML {url}", flush=config.FLUSH_LOG)

                    soup = BeautifulSoup(response.text, 'html.parser')

                    # hash contents and compare if we need to process this page
                    hash_val = hash_html_content(response.text)
                    if already_seen(seen_content_hashes, hash_val):
                        error(f"Already seen content url: {url}. No more processing done.")
                        return
                    debug(f"Adding hash {hash_val} to seen content for url: {url}")
                    seen_content_hashes.add(hash_val)

                    # Save the page
                    filename = os.path.join(output_dir, clean_url.replace('/', '_') + '.html')
                    if config.SAVE_HTML_CONTENT:
                        print(f"Save page filename: {filename}", flush=config.FLUSH_LOG)
                        save_resp_content(response, filename)
                        filename = filename.replace('.html', '.txt')
                        print(f"Save text: {filename}", flush=config.FLUSH_LOG)
                        with open(filename, 'wb') as file:
                            file.write(html_to_text(soup).encode("utf-8"))
                    else:
                        print(f"Marked page filename: {filename}", flush=config.FLUSH_LOG)

                    # Crawl internal and external links
                    child_depth = depth_effective + 1
                    if should_process_child_links(child_depth, is_peers_family, max_depth):
                        debug(f"Processing child links for {url}", flush=config.FLUSH_LOG)
                        for link in soup.find_all('a', href=True):
                            child_url = urljoin(url, link['href'])

                            # Remove the hash and the following alphanumeric (or dash) characters at the end of the string (if any)
                            child_url = re.sub(pattern_hash_url, '', child_url)

                            # if full_url not in visited and full_url <> url:
                            # Previous - if full_url not in visited:
                            if should_visit(child_url, child_depth, visited):
                                print(f"CRAWL:({depth_actual}/{depth_effective}) Parent: '{url}' Child: '{child_url}'",
                                      flush=config.FLUSH_LOG)
                                crawl(child_url, depth_actual + 1, child_depth)
                    else:
                        debug(f"Skipping of child links for {url}", flush=config.FLUSH_LOG)

                elif 'application/xml' in content_type or 'text/xml' in content_type:
                    debug(f"File appears to be XML {url}", flush=config.FLUSH_LOG)
                elif 'text/css' in content_type:
                    debug(f"File appears to be CSS {url}", flush=config.FLUSH_LOG)
                elif 'application/javascript' in content_type or 'text/javascript' in content_type:
                    debug(f"File appears to be Javascript {url}", flush=config.FLUSH_LOG)
                elif 'image/jpeg' in content_type:
                    debug(f"File appears to be JPEG image {url}", flush=config.FLUSH_LOG)
                elif 'image/png' in content_type:
                    debug(f"File appears to be PNG image {url}", flush=config.FLUSH_LOG)
                elif 'image/gif' in content_type:
                    debug(f"File appears to be GIF image {url}", flush=config.FLUSH_LOG)
                elif 'application/vnd.ms-powerpoint' in content_type:
                    debug(f"File appears to be PPT Powerpoint {url}", flush=config.FLUSH_LOG)
                else:  # Unknown type
                    debug(f"SKIPPING NOT HTML/PDF/XML: {url}", flush=config.FLUSH_LOG)
                    # endif content_type

            else:  # status code not 200
                # Handle broken link
                print(f"Broken link: {url} (Status: {response.status_code})", flush=config.FLUSH_LOG)
                archived_url = get_wayback_url(url)
                if archived_url:
                    print(f"Retrieving archived version from: {archived_url}", flush=config.FLUSH_LOG)
                    clean_url = archived_url.replace('https://', '')
                    clean_url = clean_url.replace('http://', '')
                    clean_url = clean_url.rstrip('/')
                    clean_url = clean_url.replace('?', 'QQ')
                    clean_url = clean_url.replace('=', 'EQ')
                    clean_url = clean_url.replace('&', 'AMP')
                    download_url(archived_url, os.path.join(output_dir, "archived_" + clean_url.replace('/',
                                                                                                        '_')))  # handle html or pdf?
                    # download_url(archived_url, os.path.join(output_dir, "archived_" + clean_url.replace('/', '_') + '.html'))
                    # Unless it's a PDF and not an HTML file ???
                else:
                    error(f"ERROR: No archived version found for: {url}")
                # endif for if archived
            # endif for status code 200
        except requests.exceptions.Timeout:
            error(f"ERROR: The request for {url} timed out")
        except StopIteration:
            raise  # max pages hit
        except Exception as e:
            error(f"ERROR EXCEPTION WHILE CRAWLING {url}: {e}")

    # start the initial crawl
    try:
        crawl(start_url, 0, 0)
    except StopIteration:
        print("-- Stopping iteration. Max pages hit.")
        sys.stderr.write("\n-- Stopping iteration. Max pages hit.")

    output_msg = "\n** Crawl finished, visited num pages: " + str(num_pages_visited)
    error(output_msg)  # just stderr


def main():
    # config settings
    corpus_location = config.CORPUS_FOLDER_LOCATION
    log_location = config.LOGS_FOLDER_LOCATION
    log_file = log_location + config.LOGS_NAME
    max_depth = config.MAX_DEPTH_CRAWL_LIMIT
    max_pages = config.MAX_PAGES_CRAWL_LIMIT

    # process commandline
    n = len(sys.argv)
    if n > 1:
        max_pages = int(sys.argv[1])
        print(f"max_pages override set to {max_pages}")

    # Redirect all output to log file.
    file = open(log_file, "w")  # use 'a' for append, 'w' for overwrite
    sys.stdout = file

    sys.setrecursionlimit(config.PYTHON_RECURSION_DEPTH)  # Is this truly necessary? Why wasn't 1000 enough?

    error(f"*** CRAWL SITE BEGIN")  # just stderr logging

    #crawl_site("http://www.wanttoknow.info", corpus_location, max_depth, max_pages)
    crawl_site("http://www.momentoflove.org", corpus_location, max_depth, max_pages)
    # crawl_site("https://www.wanttoknow.info/a-why-healthy-food-so-expensive-america-blame-farm-bill-congress-always-renews-make-burgers-cheaper-than-salad", "C:\\Users\\rames\\ai\\CrawlTest\\")
    # crawl_site("http://www.washingtonpost.com/wp-dyn/articles/A49449-2004Dec8.html", "C:\\Users\\marc\\ai\\CrawlTest\\")
    # crawl_site("http://martintruther.substack.com", "D:\\Dropbox\\DeepSeek\\CrawlTest\\")
    # crawl_site("https://www.newschool.edu/", "D:\\Dropbox\\DeepSeek\\CrawlTest\\")
    # crawl_site("https://explore.whatismybrowser.com/useragents/parse/", "D:\\Dropbox\\DeepSeek\\CrawlTest\\"

    error("*** CRAWL SITE END")


if __name__ == "__main__":
    main()
