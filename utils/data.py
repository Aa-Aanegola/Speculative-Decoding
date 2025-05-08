import json
import os

def load_jsonl(filepath):
    with open(filepath, "r", encoding="utf-8") as f:
        data = [json.loads(line) for line in f]
    return data