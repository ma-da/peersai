import config
import sqlite3
from datetime import datetime
from utils import clean_url, debug, error
import os
import requests

"""This module provides utility functions for caching url contents and saving metadata to a sqlite db.

Cache metadata for each downloaded URL in a SQLite database:
- URL
- File path
- File size
- Download timestamp
- (Optional: content hash or etag/header info)

Before re-downloading a file:
1. Check if the URL is in the cache.
2. If yes, confirm that the file still exists and matches expected size and/or timestamp.
3. If it matches → use local file
4. If it doesn't match → re-download and update the cache

Benefits
- Avoids redundant downloads → faster and less bandwidth-intensive
- Handles corpus changes (e.g., deleted, truncated, or stale files)
- SQLite is perfect for local metadata: portable, concurrent-safe, and fast
- You can add more metadata later: content hashes, HTTP headers, extraction success status, etc.
"""

# return clean_url, status_code (200 if okay), content_type, content, was_cached
def get_cached_content_or_request(url, headers=config.headers, timeout=15):
    cleaned_url = clean_url(url)

    cached_data = None
    if config.CACHE_ENABLED:
        cached_data = get_cached_file_content(cleaned_url, config.DB_CACHE_PATH)

    if cached_data is None:
        debug(f"cleaned_url {cleaned_url} not found in cache. Retrieving using http request...")
        response = requests.get(url, headers=headers, timeout=timeout)
        content_type = response.headers.get('Content-Type')
        return cleaned_url, response.status_code, content_type, response.content, False
    else:
        content, content_type = cached_data
        debug(f"cleaned_url {cleaned_url} was retrieved from cache")
        return cleaned_url, 200, content_type, content, True

def init_db(db_path=config.DB_CACHE_PATH):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS downloads (
            cleaned_url TEXT PRIMARY KEY,
            content_type TEXT NOT NULL,
            url_file_path TEXT NOT NULL,
            url_file_size INTEGER NOT NULL,
            text_file_path TEXT NOT NULL,
            text_file_size INTEGER NOT NULL,
            hash TEXT NOT NULL,
            download_time TEXT NOT NULL
        )
    ''')
    conn.commit()
    conn.close()

def clear_cache(db_path=config.DB_CACHE_PATH, delete_db=False):
    if delete_db and os.path.exists(db_path):
        os.remove(db_path)
        debug(f"Deleted entire cache database: {db_path}")
        return

    # Otherwise, just clear the downloads table
    if os.path.exists(db_path):
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute("DELETE FROM downloads")
        conn.commit()
        conn.close()
        debug("Cleared all entries from the downloads table.")
    else:
        debug("No cache database found.")

def update_cache(cleaned_url, content_type, url_file_path, url_file_size, text_file_path, text_file_size, hash, download_time=None, db_path=config.DB_CACHE_PATH):
    if download_time is None:
        download_time = datetime.now()
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute('''
        INSERT INTO downloads (cleaned_url, content_type, url_file_path, url_file_size, text_file_path, text_file_size, hash, download_time)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(cleaned_url) DO UPDATE SET
            url_file_path=excluded.url_file_path,
            url_file_size=excluded.url_file_size,
            text_file_path=excluded.text_file_path,
            text_file_size=excluded.text_file_size,
            download_time=excluded.download_time,
            hash=excluded.hash
    ''', (cleaned_url, content_type, url_file_path, url_file_size, text_file_path, text_file_size, hash, download_time))
    conn.commit()
    conn.close()
    debug(f"update_cached saved entry with cleaned_url {cleaned_url}, url_file {url_file_path}, url_file_size {url_file_size}, text_file_path {text_file_path}, text_file_size {text_file_size}, hash {hash}")

# Get metadata associated with cached url, returns None if not found
def get_cached_url_data(db_path, cleaned_url):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM downloads WHERE cleaned_url = ?", (cleaned_url,))
    row = cursor.fetchone()
    conn.close()
    return row

# Returns url_file_path, url_file_size, text_file_path, text_file_size
def get_cached_file_info(cleaned_url, db_path=config.DB_CACHE_PATH):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute('SELECT url_file_path, url_file_size, text_file_path, text_file_size FROM downloads WHERE cleaned_url=?', (cleaned_url,))
    row = cursor.fetchone()
    conn.close()

    if row:
        file_path, expected_size = row
        if os.path.exists(file_path):
            actual_size = os.path.getsize(file_path)
            if actual_size == expected_size:
                return row
    return None

# Returns (contents, content_type) if cached, None otherwise
def get_cached_file_content(cleaned_url, db_path=config.DB_CACHE_PATH):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute('SELECT url_file_path, url_file_size, content_type FROM downloads WHERE cleaned_url=?', (cleaned_url,))
    row = cursor.fetchone()
    conn.close()

    if row:
        file_path, expected_size, content_type = row
        if os.path.exists(file_path):
            actual_size = os.path.getsize(file_path)
            if actual_size == expected_size:
                with open(file_path, 'rb') as file:
                    contents = file.read()
                    return contents, content_type
    return None
