"""
Simple evaluation between two extraction outputs. Computes token overlap and basic stats.
"""
import json
import re
from collections import Counter

def simple_tokenize(text):
    return re.findall(r"\\w+", text.lower())

def compare_files(path_a, path_b):
    with open(path_a, 'r', encoding='utf-8') as f:
        a = json.load(f)
    with open(path_b, 'r', encoding='utf-8') as f:
        b = json.load(f)
    pages_a = {p['page']: p for p in a.get('pages', [])}
    pages_b = {p['page']: p for p in b.get('pages', [])}
    pages = sorted(set(list(pages_a.keys()) + list(pages_b.keys())))
    summary = {'pages': []}
    for p in pages:
        ta = pages_a.get(p, {}).get('text','')
        tb = pages_b.get(p, {}).get('text','')
        tokens_a = simple_tokenize(ta)
        tokens_b = simple_tokenize(tb)
        ca = Counter(tokens_a)
        cb = Counter(tokens_b)
        # overlap
        common = sum((ca & cb).values())
        total = max(1, sum(ca.values()))
        overlap_ratio = common / total
        summary['pages'].append({'page': p, 'tokens_a': len(tokens_a), 'tokens_b': len(tokens_b), 'overlap_ratio': overlap_ratio})
    # global stats
    summary['global'] = {
        'pages': len(summary['pages']),
        'mean_overlap': sum(p['overlap_ratio'] for p in summary['pages']) / max(1, len(summary['pages']))
    }
    return summary
