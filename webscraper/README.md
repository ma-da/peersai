# peers ai webscraper
The webscraper for downloading peers-family corpus training material

See config settings in config.py before running.
- you will likely want to change MAX_PAGES_CRAWL_LIMIT (set to 0 for no limit)

To run the web crawler:
./run.sh [optional max number of pages to crawl, omit to crawl up to config limit]

To only regenerate missing txt files from corpus files:
./regen_missing_txt.sh

To regenerate all missing txt files from corpus files:
./regen_all_txt.sh

