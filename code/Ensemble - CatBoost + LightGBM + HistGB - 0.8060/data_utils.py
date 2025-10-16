import json
from typing import Dict, Any, Generator

def read_jsonl(path: str):
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            yield json.loads(line)

def safe_get(d: Dict, *keys, default=None):
    cur = d
    for k in keys:
        if cur is None:
            return default
        cur = cur.get(k, None) if isinstance(cur, dict) else None
    return cur if cur is not None else default