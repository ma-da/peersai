#!/usr/bin/env python3

"""
 TITLE: PEERS AI Project Website Scraper (WantToKnowScraper)
 VERSION: 1.0
 AUTHOR: Marc Baber
 DATE: 06 MAR 2025

 TO DO LIST:
 1. Make FIFO stack, for breadth first, not depth
 2. No hash suffixes
 3. Don't take HREFs from archived docs
 4. Need a way to detect and avoid recursion. PRUNE???
 5. If file already local, load it instead???
 6. Prune out MK docs


 1. Fix MomentOfLove navbar recursions on g/victim_or_creator_vs and inspiration/inspiring-videos
 2. Set recursion limit higher (apparently 1000 wasn't high enough). But is there a way to avoid getting so deep?
    I might have to go breadth first (instead of depth first) using a FIFO set/stack
    a. seed the FIFO with just the first (base) URL
    b. WHILE there's anything in the FIFO queue that has not already been visited:
       crawl the first URL:
       retrieve file (html or pdf) -- getting archive if necessary
       make .txt file
       If HTML, add all hrefs to FIFO stack, but not if archive (push right)
       Add this url to visited.
       WEND
 3. Don't visit a # hash href if the base URL has already been visited
 4. Why did processing stop after 50 hrs on a PPT file: http://www.cs.cmu.edu/~pausch/Randy/Randy/pauschlastlecture.ppt ?
    (update: did not crash, just stalled)
 5.

 Don't prune page tree while still on "home site" which should include whole family of PEERS
 Websites and online courses currently managed by PEERS:

 www.momentoflove.org - Every person in the world has a heart
 www.weboflove.org - Strengthening the web of love that interconnects us all
 www.WantToKnow.info - Revealing major cover-ups and working together for a brighter future
 www.newsarticles.media - Collection of under-reported major media news articles
 www.divinemystery.net - Mystical musings of a spiritual explorer
 www.inspiringcommunity.org - Building a global community for all
 www.wisdomcourses.net - Free online courses inspire you to greatness
 www.inspirationcourse.net - The Inspiration Course: Opening to more love and deeper connection
 www.hidden-knowledge.net - Hidden Knowledge Course: Illuminating shadow aspects of our world
 www.insightcourse.net - The Insight Course: The best of the Internet all in one free course
 www.transformationteam.net - Transformation Team: Building bridges to expanded consciousness
 www.gatheringspot.net - Dynamic community networking portal for course graduates
"""
import cache
import config
from pybloom_live import BloomFilter

import content_filter
import pdf_fetcher
import os
import re
import fitz  # PyMuPDF
import sys
from web_scraper_base import *

import utils
from utils import *
from bs4 import BeautifulSoup
from urllib.parse import urljoin


def crawl_site(start_url, output_dir, max_depth=2, max_pages=-1):
    init_working_dirs(output_dir)

    # visited tracks the urls we have visited
    visited = set()

    # seen_content_hashes tracks the hashed value of the url contents to see if we have seen it before
    seen_content_hashes = BloomFilter(capacity=1_000_000, error_rate=0.00001)
    num_pages_visited = 0

    # crawl should process only already filtered urls
    def crawl(url, depth_actual, depth_effective):
        nonlocal num_pages_visited

        if num_pages_visited > 0 and num_pages_visited % config.PROGRESS_REPORT_N_PAGES == 0:
            error(f"Webcrawler crawled {num_pages_visited} number of pages.")

        is_peers_family = False
        if config.pattern_peers_family.match(url):
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
            cleaned_url, status_code, content_type, content, was_cached = cache.get_cached_content_or_request(url, headers=config.headers, timeout=15)
            if status_code == 200:
                if config.ENABLE_PROCESS_PDFS and 'application/pdf' in content_type:
                    print(f"File appears to be PDF {url}", flush=config.FLUSH_LOG)

                    pdf_output_path = os.path.join(output_dir, cleaned_url.replace('/', '_') + '.pdf')
                    pdf_output_txt_filename = pdf_output_path.replace('.pdf', '.txt')
                    print(f"Save PDF-to-text: {pdf_output_txt_filename}", flush=config.FLUSH_LOG)

                    # Step 1: Download the PDF
                    pdf_fetcher.download_pdf(url, pdf_output_path)

                    # Step 2: Extract text from the PDF
                    title, extracted_text = pdf_fetcher.extract_text_from_pdf(pdf_output_path)

                    # Step 3: Save the extracted text to a file
                    pdf_fetcher.save_text_to_file(title, extracted_text, pdf_output_txt_filename)

                elif 'text/html' in content_type:
                    debug(f"File appears to be HTML {url}", flush=config.FLUSH_LOG)
                    soup = BeautifulSoup(content, 'html.parser')

                    # hash contents and compare if we need to process this page
                    hash_val = hash_html_content(content)
                    if already_seen(seen_content_hashes, hash_val):
                        debug(f"Already seen content url: {url}. No more processing done.")
                        return
                    debug(f"Adding hash {hash_val} to seen content for url: {url}")
                    seen_content_hashes.add(hash_val)

                    # Save the page to the corpus
                    if not was_cached:
                        debug(f"Processing url {cleaned_url} for corpus collection")
                        filename = os.path.join(output_dir, cleaned_url.replace('/', '_') + '.html')
                        if config.SAVE_HTML_CONTENT:
                            # save html content
                            debug(f"Save page filename: {filename}", flush=config.FLUSH_LOG)
                            save_resp_content(content, filename)

                            txt_filename = filename.replace('.html', '.txt')
                            utils.save_txt_content_to_file(txt_filename, content)

                            # save cache metadata entry
                            if config.CACHE_ENABLED:
                                url_file_size = os.path.getsize(filename)
                                txt_file_size = os.path.getsize(txt_filename)
                                cache.update_cache(cleaned_url, 'text/html', filename, url_file_size, txt_filename, txt_file_size, hash_val)
                        else:
                            debug(f"Marked page filename: {filename}", flush=config.FLUSH_LOG)
                    else:
                        # if cached, lets check if we need to regenerate the txt file if it doesn't exist
                        txt_filename = os.path.join(output_dir, cleaned_url.replace('/', '_') + '.txt')
                        if not os.path.exists(txt_filename):
                            error(f"Regenerating {txt_filename}")
                            utils.save_txt_content_to_file(txt_filename, content)
                        else:
                            debug(f"Not Regenerating {txt_filename}")
                        debug(f"Skipped url {cleaned_url} for corpus collection because it was already in cache.")


                    # Crawl internal and external links
                    child_depth = depth_effective + 1
                    if should_process_child_links(child_depth, is_peers_family, max_depth):
                        debug(f"Processing child links for {url}", flush=config.FLUSH_LOG)
                        for link in soup.find_all('a', href=True):
                            child_url = urljoin(url, link['href'])

                            # Remove the hash and the following alphanumeric (or dash) characters at the end of the string (if any)
                            child_url = re.sub(config.pattern_hash_url, '', child_url)

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
                print(f"Broken link: {url} (Status: {status_code})", flush=config.FLUSH_LOG)
                archived_url = get_wayback_url(url)
                if archived_url:
                    print(f"Retrieving archived version from: {archived_url}", flush=config.FLUSH_LOG)
                    cleaned_url = archived_url.replace('https://', '')
                    cleaned_url = cleaned_url.replace('http://', '')
                    cleaned_url = cleaned_url.rstrip('/')
                    cleaned_url = cleaned_url.replace('?', 'QQ')
                    cleaned_url = cleaned_url.replace('=', 'EQ')
                    cleaned_url = cleaned_url.replace('&', 'AMP')
                    download_url(archived_url, os.path.join(output_dir, "archived_" + cleaned_url.replace('/',
                                                                                                        '_')))  # handle html or pdf?
                    #download_url(archived_url, os.path.join(output_dir, "archived_" + clean_url.replace('/', '_') + '.html'))
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
    cache.init_db()

    if config.FLUSH_CACHE_ON_START:
        cache.clear_cache()

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
