"""
 PEERS AI Project Multi-threaded Website Scraper (WantToKnowScraper)
 The following is an implementation of the webscraper that is fully multi-threaded.
"""
import queue
import threading
import time

import cache
import config
from pybloom_live import BloomFilter

import content_filter
import pdf_fetcher
import os
import re
import fitz  # PyMuPDF
import sys

import utils
import web_scraper_old
from utils import *
from bs4 import BeautifulSoup
from urllib.parse import urljoin
from queue import Queue
from web_scraper_base import *

# Set the number of multithreaded workers here
NUM_WORKERS = 8

# Multithreaded version of crawl_site()
def crawl_site(start_url, output_dir, max_depth=2, max_pages=-1, refresh_queue=True):
    init_working_dirs(output_dir)

    # visited tracks the urls we have visited
    visited = set()
    visited_lock = threading.Lock()

    # seen_content_hashes tracks the hashed value of the url contents to see if we have seen it before
    seen_content_hashes = BloomFilter(capacity=1_000_000, error_rate=0.00001)
    seen_content_hashes_lock = threading.Lock()

    num_pages_visited = 0
    num_pages_visited_lock = threading.Lock()
    next_num_pages_visited_report = config.PROGRESS_REPORT_N_PAGES

    # holds the urls we have not yet visited
    url_queue = Queue()
    stop_event = threading.Event()


    def stop_crawl(id):
        error(f"Worker {id} signaled to stop the crawl")
        stop_event.set()


    def worker(id):
        debug(f"Worker id {id} started")
        while True:
            if stop_event.is_set():
                debug(f"Worker id got stop event{id}")
                break

            try:
                url, depth_actual, depth_effective = url_queue.get(timeout=1)
            except queue.Empty:
                # error(f"Worker {id} got empty")
                time.sleep(1.0)
                continue

            try:
                if url is None:
                    debug(f"Worker id got queue shutdown {id}")
                    break

                debug(f"Worker {id} got work {url}")

                if not stop_event.is_set():
                    crawl(url, depth_actual, depth_effective)

            except StopIteration:
                # only need to output this once
                if not stop_event.is_set():
                    error("-- Stopping iteration. Max pages hit.")
                    sys.stderr.write("\n-- Stopping iteration. Max pages hit.")
                break
            except Exception as e:
                error(f"Worker id {id} got runtime exception: {e}")
                break
            finally:
                url_queue.task_done()

        if not stop_event.is_set():
            stop_crawl(id)

        debug(f"Finished worker id {id}")

    def add_url_to_crawl(url, depth_actual, depth_effective):
        url_queue.put((url, depth_actual, depth_effective))
        cache.save_pending_url_to_db(url, depth_actual, depth_effective)
        debug(f"Adding url_to_crawl: {url}")

    def finalize_url_to_crawl(url):
        cache.delete_pending_url_from_db(url)
        debug(f"Finalized url_to_crawl: {url}")

    def queue_join_with_timeout(timeout=1.0):
        nonlocal num_pages_visited
        nonlocal next_num_pages_visited_report

        """Wait for queue to be empty or stop_event to be set."""
        while url_queue.unfinished_tasks > 0:
            with num_pages_visited_lock:
                if num_pages_visited > 0 and num_pages_visited > next_num_pages_visited_report:
                    error(f"Webcrawler crawled {num_pages_visited} number of pages.")
                    next_num_pages_visited_report = next_num_pages_visited_report + config.PROGRESS_REPORT_N_PAGES

            if stop_event.is_set():
                break

            time.sleep(timeout)


    # crawl should process only already filtered urls
    def crawl(url, depth_actual, depth_effective):
        nonlocal num_pages_visited

        is_peers_family = False
        if config.pattern_peers_family.match(url):
            depth_effective = 0  # Effective depth is how many hops from home domain(s)
            is_peers_family = True
            debug(f"URL is in Home Domain(s): {url}", flush=config.FLUSH_LOG)
        else:
            debug(f"URL is NOT in Home Domain(s) {url}", flush=config.FLUSH_LOG)

        with num_pages_visited_lock:
            num_pages_visited = num_pages_visited + 1
            if max_pages > 0 and num_pages_visited > max_pages:
                print("Maximum pages hit. Stopping crawl.")
                raise StopIteration("Maximum pages hit")

        debug(f"Adding {url} to visited set")
        visited.add(url)

        print(f"({depth_actual}/{depth_effective}) CRAWLING: {url}", flush=config.FLUSH_LOG)

        # Try to fetch the page
        try:
            # original version
            # cleaned_url, status_code, content_type, content, was_cached = cache.get_cached_content_or_request(url, headers=config.headers, timeout=15)
            # playwright version
            cleaned_url, status_code, content_type, content, was_cached = cache.get_cached_content_or_playwright_request(url, headers=config.headers, timeout=60000)
            if status_code == 200:
                if config.ENABLE_PROCESS_PDFS and 'application/pdf' in content_type:
                    print(f"File appears to be PDF {url}", flush=config.FLUSH_LOG)

                    pdf_output_path = os.path.join(output_dir, cleaned_url.replace('/', '_') + '.pdf')
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
                    soup = BeautifulSoup(content, 'html.parser')
                    utils.body_adjustments(soup)
                    soup_content = soup.prettify().encode()

                    # hash contents and compare if we need to process this page
                    hash_val = hash_html_content(soup_content)
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
                            debug(f"SAVE page filename: {filename}", flush=config.FLUSH_LOG)
                            save_resp_content(soup_content, filename)

                            txt_filename = filename.replace('.html', '.txt')
                            utils.save_txt_content_to_file(txt_filename, soup_content)

                            # save cache metadata entry
                            if config.CACHE_ENABLED:
                                url_file_size = os.path.getsize(filename)
                                txt_file_size = os.path.getsize(txt_filename)
                                cache.update_cache(cleaned_url, 'text/html', filename, url_file_size, txt_filename, txt_file_size, hash_val)
                        else:
                            debug(f"MARK page filename: {filename}", flush=config.FLUSH_LOG)
                    else:
                        # if cached, lets check if we need to regenerate the txt file if it doesn't exist
                        txt_filename = os.path.join(output_dir, cleaned_url.replace('/', '_') + '.txt')
                        if not os.path.exists(txt_filename):
                            error(f"Regenerating {txt_filename}")
                            utils.save_txt_content_to_file(txt_filename, soup_content)
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
                                print(f"ADD_TO_CRAWL:({depth_actual}/{depth_effective}) Parent: '{url}' Child: '{child_url}'",
                                      flush=config.FLUSH_LOG)
                                add_url_to_crawl(child_url, depth_actual + 1, child_depth)
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
    #try:
    #    crawl(start_url, 0, 0)
    #except StopIteration:
    #    print("-- Stopping iteration. Max pages hit.")
    #    sys.stderr.write("\n-- Stopping iteration. Max pages hit.")

    # loads any pending urls from the last run that haven't been processed yet
    if refresh_queue:
        cache.load_pending_urls_from_db(url_queue)
        qsize = url_queue.qsize()
        if qsize > 0:
            error(f"Pending url_queue was refreshed with {qsize} elements")
            cache.clear_pending_url_queue_db()

    # adds the initial seed url to crawl
    add_url_to_crawl(start_url, 0, 0)

    id = 1
    threads = []
    error(f"Starting {NUM_WORKERS} number of worker threads...")
    for _ in range(NUM_WORKERS):
        debug(f"Starting worker thread {id}...")
        t = threading.Thread(target=worker, args=(id, ), daemon=False)
        t.start()
        id = id + 1
        threads.append(t)

    error("Waiting for url_queue to join...")
    queue_join_with_timeout()

    stop_event.set()

    error("Waiting for worker threads to join...")
    for t in threads:
        t.join()

    error("Workers completed.")

    output_msg = "\n** Parallel web crawl finished, visited num pages: " + str(num_pages_visited)
    error(output_msg)  # just stderr


def main():
    cache.init_db()

    if config.FLUSH_CACHE_ON_START:
        cache.clear_cache()

    # start_url
    start_url = "http://www.momentoflove.org"

    # config settings
    corpus_location = config.CORPUS_FOLDER_LOCATION
    log_location = config.LOGS_FOLDER_LOCATION
    log_file = log_location + config.LOGS_NAME
    max_depth = config.MAX_DEPTH_CRAWL_LIMIT
    max_pages = config.MAX_PAGES_CRAWL_LIMIT
    refresh_queue = config.LOAD_PENDING_QUEUE_ON_START

    # Redirect all output to log file.
    file = open(log_file, "w")  # use 'a' for append, 'w' for overwrite
    sys.stdout = file

    # process commandline
    # Optional 1st arg – start crawl with given url
    # Optional 2nd arg - max pages to crawl
    n = len(sys.argv)
    error(f"CMD # {n}")
    if n > 1:
        cmd_url = sys.argv[1]
        if not cmd_url.startswith("http://") and not cmd_url.startswith("https://"):
            cmd_url = "http://" + cmd_url
        start_url = cmd_url
        refresh_queue = False
        error(f"Override start url {start_url}")

    if n > 2:
        max_pages_cmd = int(sys.argv[2])
        error(f"Found new max_pages to crawl setting {max_pages_cmd}")
        max_pages = max_pages_cmd

    sys.setrecursionlimit(config.PYTHON_RECURSION_DEPTH)  # Is this truly necessary? Why wasn't 1000 enough?

    error(f"*** CRAWL SITE BEGIN at url: {start_url}")  # just stderr logging

    crawl_site(start_url, corpus_location, max_depth, max_pages, refresh_queue)

    #crawl_site("http://www.wanttoknow.info", corpus_location, max_depth, max_pages)
    #crawl_site("http://www.momentoflove.org", corpus_location, max_depth, max_pages)

    #crawl_site("http://www.momentoflove.org", corpus_location, max_depth, max_pages)
    #crawl_site("http://www.weboflove.org", corpus_location, max_depth, max_pages)
    #crawl_site("http://www.newsarticles.media", corpus_location, max_depth, max_pages)
    #crawl_site("http://www.divinemystery.net", corpus_location, max_depth, max_pages)
    #crawl_site("http://www.inspiringcommunity.org", corpus_location, max_depth, max_pages)
    #crawl_site("http://www.wisdomcourses.net", corpus_location, max_depth, max_pages)
    #crawl_site("http://www.inspirationcourse.net", corpus_location, max_depth, max_pages)
    #crawl_site("http://www.insightcourse.net", corpus_location, max_depth, max_pages)
    #crawl_site("http://www.transformationteam.net", corpus_location, max_depth, max_pages)
    #crawl_site("http://www.gatheringspot.net", corpus_location, max_depth, max_pages)
    # crawl_site("https://www.wanttoknow.info/a-why-healthy-food-so-expensive-america-blame-farm-bill-congress-always-renews-make-burgers-cheaper-than-salad", "C:\\Users\\rames\\ai\\CrawlTest\\")
    # crawl_site("http://www.washingtonpost.com/wp-dyn/articles/A49449-2004Dec8.html", "C:\\Users\\marc\\ai\\CrawlTest\\")
    # crawl_site("http://martintruther.substack.com", "D:\\Dropbox\\DeepSeek\\CrawlTest\\")
    # crawl_site("https://www.newschool.edu/", "D:\\Dropbox\\DeepSeek\\CrawlTest\\")
    # crawl_site("https://explore.whatismybrowser.com/useragents/parse/", "D:\\Dropbox\\DeepSeek\\CrawlTest\\"

    error("*** CRAWL SITE END")


if __name__ == "__main__":
    main()
