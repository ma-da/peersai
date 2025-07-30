import os
import re

import config
import utils
from utils import *

# Check if this url should be processed
# Note preferable check this prior to recursive call of crawl
def should_visit(url, depth_effective, visited_set):
    debug(f"Should visit url {url} depth {depth_effective}?", flush=config.FLUSH_LOG)

    if config.pattern_filter_list.search(url):
        debug(f"Visit declined. Skipping pattern filter list {url}", flush=config.FLUSH_LOG)
        return False

    if config.pattern_archive_url.match(url):
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

    # don't process substack comments
    if utils.is_substack_comment_page(url):
        debug(f"Visit declined. Referred to substack comment: {url}", flush=config.FLUSH_LOG)
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


# make sure all artifact dirs exists
def init_working_dirs(output_dir):
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(config.LOGS_FOLDER_LOCATION, exist_ok=True)
    os.makedirs(config.DB_CACHE_LOCATION, exist_ok=True)

