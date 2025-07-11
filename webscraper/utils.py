#!/usr/bin/env python3

from bs4 import BeautifulSoup
import config
import hashlib
import requests
import sys


# print helper that only prints if debug enabled. same args as print()
def debug(*args, **kwargs):
    if config.DEBUG_ENABLED:
        print(*args, **kwargs)

def error(*args, **kwargs):
    print(*args, file=sys.stderr, flush=True, **kwargs)

def download_url(url, path):
    response = requests.get(url, headers=config.headers)
    with open(path, 'wb') as file:
        file.write(response.content)

# save the response contents from requests into a file
def save_resp_content(resp, filename):
    with open(filename, 'wb') as file:
        file.write(resp.content)


def get_wayback_url(url):
    """Fetch the closest archived version of a URL from Archive.org."""
    params = {"url": url}
    response = requests.get(config.WAYBACK_API, params=params)
    if response.status_code == 200:
        result = response.json()
        if "archived_snapshots" in result and "closest" in result["archived_snapshots"]:
            return result["archived_snapshots"]["closest"]["url"]
    else:
        print(f"No archived snapshot found: {url}", flush=config.FLUSH_LOG)
    return None


def html_to_text(mysoup):
    # print(f"Enter html_to_text", flush=config.FLUSH_LOG)

    # Remove script and style elements
    for script_or_style in mysoup(['script', 'style']):
        script_or_style.decompose()

    # Extract text
    text = mysoup.get_text(separator=' ')

    # Clean up whitespace
    lines = (line.strip() for line in text.splitlines())  # Remove leading/trailing whitespace
    chunks = (phrase.strip() for line in lines for phrase in line.split("  "))  # Split multi-spaces
    text = ' '.join(chunk for chunk in chunks if chunk)  # Join non-empty chunks

    # print(f"Exit html_to_text", flush=config.FLUSH_LOG)
    return text


def hash_html_content(html):
    normalized = html.strip().lower()
    return hashlib.sha256(normalized.encode('utf-8')).hexdigest()


def hash_soup(soup: BeautifulSoup) -> str:
    html = soup.pretty()
    return hash_html_content(html)


def already_seen(filter, content_hash):
    return content_hash in filter
