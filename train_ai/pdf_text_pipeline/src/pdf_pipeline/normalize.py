"""
Normalization utilities:
- unicode normalization
- remove headers/footers heuristics
- join lines while preserving hyphens and lists
- produce 'clean' text and an annotated JSON output
"""
import unicodedata
import re
import json

def normalize_unicode(s):
    # Use NFKC to fold compatibility forms but keep semantic chars
    return unicodedata.normalize('NFKC', s)

def strip_control_chars(s):
    return ''.join(ch for ch in s if ch == '\n' or (ch.isprintable() and ord(ch) != 0x0c))

def remove_headers_footers(lines, header_footer_freq=0.2):
    # naive: find repeated lines near top/bottom across pages
    # Input: list of pages, each page is dict with 'lines' (list)
    from collections import Counter, defaultdict
    top_counter = Counter()
    bottom_counter = Counter()
    for p in lines:
        if len(p['lines']) == 0:
            continue
        top_counter[p['lines'][0]] += 1
        bottom_counter[p['lines'][-1]] += 1
    n_pages = len(lines)
    top_common = {ln for ln,c in top_counter.items() if c / n_pages >= header_footer_freq}
    bottom_common = {ln for ln,c in bottom_counter.items() if c / n_pages >= header_footer_freq}
    # remove if matches
    out = []
    for p in lines:
        lns = p['lines'][:]
        if lns and lns[0] in top_common:
            lns = lns[1:]
        if lns and lns[-1] in bottom_common:
            lns = lns[:-1]
        out.append({'page': p['page'], 'lines': lns})
    return out

def join_lines_preserve_structure(text):
    # heuristic: join lines preserving paragraphs. Handles hyphenation.
    lines = [ln.rstrip() for ln in text.splitlines()]
    out_lines = []
    buf = ''
    for ln in lines:
        if not ln:
            if buf:
                out_lines.append(buf.strip())
                buf = ''
            continue
        # detect list item or big indent -> preserve newline
        if re.match(r'\s*[-\*\u2022\d+\.)]', ln):
            if buf:
                out_lines.append(buf.strip())
                buf = ''
            out_lines.append(ln.strip())
            continue
        # hyphenation join
        if buf.endswith('-'):
            buf = buf[:-1] + ln.lstrip()
        else:
            if buf:
                buf = buf + ' ' + ln.lstrip()
            else:
                buf = ln
    if buf:
        out_lines.append(buf.strip())
    return '\n\n'.join(out_lines)

def normalize_intermediate_json(input_json_path, output_json_path, remove_headers=True):
    with open(input_json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    pages = data.get('pages', [])
    pages_lines = []
    for p in pages:
        txt = p.get('text','')
        txt = normalize_unicode(txt)
        txt = strip_control_chars(txt)
        # split into lines for header/footer removal heuristics
        lines = [l for l in txt.splitlines()]
        pages_lines.append({'page': p['page'], 'lines': lines, 'source': p.get('source'), 'confidence': p.get('confidence',0)})
    if remove_headers:
        pages_lines = remove_headers_footers(pages_lines)
    # join lines back using the smarter joiner
    out_pages = []
    for p in pages_lines:
        joined = join_lines_preserve_structure('\n'.join(p['lines']))
        out_pages.append({'page': p['page'], 'text': joined, 'source': p['source'], 'confidence': p['confidence']})
    with open(output_json_path, 'w', encoding='utf-8') as f:
        json.dump({'pdf_path': data.get('pdf_path'), 'pages': out_pages}, f, ensure_ascii=False, indent=2)
    return output_json_path
