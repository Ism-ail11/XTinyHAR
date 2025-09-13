import os, json
import numpy as np
from typing import Any, Dict

def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)

def save_npz(path: str, **arrays):
    np.savez_compressed(path, **arrays)

def append_jsonl(path: str, obj: Dict[str, Any]):
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(obj) + "\n")
