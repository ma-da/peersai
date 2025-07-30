import sys
import config


def main():
    n = len(sys.argv)
    if n == 1:
        print("Please input a url to check")
        return

    url = sys.argv[1]
    if not url.startswith("http://") and not url.startswith("https://"):
        url = "http://" + url

    print(f"Checking url {url}")
    if config.pattern_peers_family.search(url):
        print(f"URL is in Home Domain(s): {url}")
    else:
        print(f"URL is NOT in Home Domain(s) {url}")

    if config.pattern_filter_list.search(url):
        print(f"URL was FILTERED: {url}")

if __name__ == "__main__":
    main()
