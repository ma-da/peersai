# this file is a test sandbox for testing urls

import requests
from playwright.sync_api import sync_playwright

url = "https://www.martintruther.com/archives"
headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/115 Safari/537.36"
}

print("\n-- REQUESTS --")
try:
    resp = requests.get(url, headers=headers, timeout=30)
    print("Status Code:", resp.status_code)
    print("Content-Type:", resp.headers.get("Content-Type"))
    print("Body snippet:", resp.text[:500])
except Exception as e:
    print("Request error:", repr(e))

def fetch_with_playwright(url):
    with sync_playwright() as pw:
        browser = pw.chromium.launch(headless=True)
        page = browser.new_page()
        resp = page.goto(url, timeout=60000)
        page.wait_for_load_state("networkidle", timeout=60000)
        return resp.status, page.content()

print("\n-- PLAYWRIGHT --")
try:
    status, html = fetch_with_playwright("https://www.martintruther.com/archives")
    print(status)
    print(html[:500])
except Exception as e:
    print("Playwright request error:", repr(e))
