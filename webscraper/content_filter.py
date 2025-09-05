from newspaper import Article

import utils

# This strategy extracts the content using newspaper3k
def extract_content_newspaper(html_content):
    article = Article(url="https://dummy.com")  # dummy url
    article.set_html(html_content)
    article.parse()
    return article.title, article.text


# This strategy strips out the html tags
def extract_content_from_soup(soup):
    return utils.html_to_text(soup)
