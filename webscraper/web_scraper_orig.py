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
# 5. If file already local, load it insteaed???
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

import os
import requests
import re
import fitz  # PyMuPDF
import sys
from bs4 import BeautifulSoup
from urllib.parse import urljoin

# Wayback Machine API endpoint
WAYBACK_API = "http://archive.org/wayback/available"
headers = {"User-Agent": "AiBot/1.0"}

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
        # momentoflove\.org        | 
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


def download_pdf(url, output_path):
    """Download a PDF file from a URL and save it locally."""
    response = requests.get(url)
    if response.status_code == 200:
        with open(output_path, 'wb') as file:
            file.write(response.content)
        print(f"PDF downloaded and saved to {output_path}", flush=True)
    else:
        print(f"Failed to download PDF. Status code: {response.status_code}", flush=True)


def download_url(url, path):
    response = requests.get(url, headers=headers)
    with open(path, 'wb') as file:
        file.write(response.content)


def get_wayback_url(url):
    """Fetch the closest archived version of a URL from Archive.org."""
    params = {"url": url}
    response = requests.get(WAYBACK_API, params=params)
    if response.status_code == 200:
        result = response.json()
        if "archived_snapshots" in result and "closest" in result["archived_snapshots"]:
            return result["archived_snapshots"]["closest"]["url"]
    else:
        print(f"No archived snapshot found: {url}", flush=True)
    return None


def html_to_text(mysoup):
    # print(f"Enter html_to_text", flush=True)

    # Remove script and style elements
    for script_or_style in mysoup(['script', 'style']):
        script_or_style.decompose()

    # Extract text
    text = mysoup.get_text(separator=' ')

    # Clean up whitespace
    lines = (line.strip() for line in text.splitlines())  # Remove leading/trailing whitespace
    chunks = (phrase.strip() for line in lines for phrase in line.split("  "))  # Split multi-spaces
    text = ' '.join(chunk for chunk in chunks if chunk)  # Join non-empty chunks

    # print(f"Exit html_to_text", flush=True)
    return text


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
        print("Text extracted successfully.", flush=True)
    except Exception as e:
        print(f"Error extracting text: {e}", flush=True)
    return text


def crawl_site(start_url, output_dir, max_depth=2):
    os.makedirs(output_dir, exist_ok=True)
    visited = set()

    def crawl(url, depth_actual, depth_effective):

        if pattern_peers_family.match(url):
            depth_effective = 0  # Effective depth is how many hops from home domain(s)
            print(f"URL is in Home Domain(s): {url}", flush=True)
        else:
            print(f"URL is NOT in Home Domain(s) {url}", flush=True)

        if pattern_archive_url.match(url):
            print(f"SKIPPING Archive URL {url}", flush=True)
            return

        if url in visited:
            print(f"Previously visited: {url}", flush=True)
            return
        print(f"Adding {url} to visited set")
        visited.add(url)

        # don't stray too far from home domain(s)
        if depth_effective > max_depth:
            print(f"Effective depth exceeded {depth_effective}", flush=True)
            return

            # Don't process image files
        if (bool(re.search('.jpe+g$', url)) or bool(re.search('.gif$', url)) or bool(re.search('.png$', url))):
            print(f"SKIPPING IMAGE: {url}", flush=True)
            return

        # Don't process mailto's
        if (bool(re.search('^mailto:', url))):
            print(f"SKIPPING MAILTO: {url}", flush=True)
            return

        # print(f"({depth_actual}/{depth_effective}) CRAWLING: {url}", flush=True)

        # Try to fetch the page
        try:
            response = requests.get(url, headers=headers, timeout=15)
            if response.status_code == 200:
                clean_url = url.replace('https://', '')
                clean_url = clean_url.replace('http://', '')
                clean_url = clean_url.rstrip('/')

                content_type = response.headers.get('Content-Type')

                if 'application/pdf' in content_type:

                    print(f"File appears to be PDF {url}", flush=True)

                    pdf_output_path = os.path.join(output_dir, clean_url.replace('/', '_') + '.pdf')
                    pdf_output_txt_filename = pdf_output_path.replace('.pdf', '.txt')
                    print(f"Save PDF-to-text: {pdf_output_txt_filename}", flush=True)

                    # Step 1: Download the PDF
                    download_pdf(url, pdf_output_path)

                    # Step 2: Extract text from the PDF
                    extracted_text = extract_text_from_pdf(pdf_output_path)

                    # Step 3: Save the extracted text to a file
                    save_text_to_file(extracted_text, pdf_output_txt_filename)

                elif 'text/html' in content_type:

                    print(f"File appears to be HTML {url}", flush=True)
                    soup = BeautifulSoup(response.text, 'html.parser')
                    filename = os.path.join(output_dir, clean_url.replace('/', '_') + '.html')
                    # Save the page
                    print(f"Save page filename: {filename}", flush=True)
                    download_url(url, filename)
                    filename = filename.replace('.html', '.txt')
                    print(f"Save text: {filename}", flush=True)
                    with open(filename, 'wb') as file:
                        file.write(html_to_text(soup).encode("utf-8"))

                    # Crawl internal and external links
                    for link in soup.find_all('a', href=True):
                        full_url = urljoin(url, link['href'])

                        # Remove the hash and the following alphanumeric (or dash) characters at the end of the string (if any)
                        full_url = re.sub(pattern_hash_url, '', full_url)

                        # if full_url not in visited and full_url <> url:
                        if full_url not in visited:
                            print(f"CRAWL:({depth_actual}/{depth_effective}) Parent: {url} Child: {full_url}",
                                  flush=True)
                            crawl(full_url, depth_actual + 1, depth_effective + 1)

                elif 'application/xml' in content_type or 'text/xml' in content_type:
                    print(f"File appears to be XML {url}", flush=True)
                elif 'text/css' in content_type:
                    print(f"File appears to be CSS {url}", flush=True)
                elif 'application/javascript' in content_type or 'text/javascript' in content_type:
                    print(f"File appears to be Javascript {url}", flush=True)
                elif 'image/jpeg' in content_type:
                    print(f"File appears to be JPEG image {url}", flush=True)
                elif 'image/png' in content_type:
                    print(f"File appears to be PNG image {url}", flush=True)
                elif 'image/gif' in content_type:
                    print(f"File appears to be GIF image {url}", flush=True)
                elif 'application/vnd.ms-powerpoint' in content_type:
                    print(f"File appears to be PPT Powerpoint {url}", flush=True)
                else:  # Unknown type
                    print(f"SKIPPING NOT HTML/PDF/XML: {url}", flush=True)
                    # endif content_type

            else:  # status code not 200
                # Handle broken link
                print(f"Broken link: {url} (Status: {response.status_code})", flush=True)
                archived_url = get_wayback_url(url)
                if archived_url:
                    print(f"Retrieving archived version from: {archived_url}", flush=True)
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
                    print(f"ERROR: No archived version found for: {url}", flush=True)
                # endif for if archived
            # endif for status code 200
        except requests.exceptions.Timeout:
            print(f"ERROR: The request for {url} timed out", flush=True)
        except Exception as e:
            print(f"ERROR EXCEPTION WHILE CRAWLING {url}: {e}", flush=True)

    crawl(start_url, 0, 0)


# Redirect all output to log file.
file = open("C:\\Users\\rames\\ai\\CrawlTest\\WantToKnowScraper.log", "a")
sys.stdout = file

sys.setrecursionlimit(2000)  # Is this truly necessary? Why wasn't 1000 enough?

print(f"CRAWL SITE BEGIN", flush=True)

crawl_site("http://www.wanttoknow.info", "C:\\Users\\rames\\ai\\CrawlTest\\")
# crawl_site("https://www.wanttoknow.info/a-why-healthy-food-so-expensive-america-blame-farm-bill-congress-always-renews-make-burgers-cheaper-than-salad", "C:\\Users\\rames\\ai\\CrawlTest\\")
# crawl_site("http://www.washingtonpost.com/wp-dyn/articles/A49449-2004Dec8.html", "C:\\Users\\marc\\ai\\CrawlTest\\")
# crawl_site("http://martintruther.substack.com", "D:\\Dropbox\\DeepSeek\\CrawlTest\\")
# crawl_site("https://www.newschool.edu/", "D:\\Dropbox\\DeepSeek\\CrawlTest\\")
# crawl_site("https://explore.whatismybrowser.com/useragents/parse/", "D:\\Dropbox\\DeepSeek\\CrawlTest\\")

print(f"CRAWL SITE END", flush=True)
