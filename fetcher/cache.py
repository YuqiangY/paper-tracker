from __future__ import annotations
import hashlib
import json
import logging
import os
import time

log = logging.getLogger(__name__)


class DiskCache:
    def __init__(self, cache_dir: str, ttl_seconds: int = 86400):
        self.cache_dir = cache_dir
        self.ttl = ttl_seconds
        os.makedirs(cache_dir, exist_ok=True)

    def _key_path(self, key: str) -> str:
        h = hashlib.sha256(key.encode()).hexdigest()[:16]
        return os.path.join(self.cache_dir, f"{h}.json")

    def get(self, key: str):
        path = self._key_path(key)
        if not os.path.exists(path):
            return None
        try:
            with open(path) as f:
                data = json.load(f)
            if time.time() - data["ts"] > self.ttl:
                os.unlink(path)
                return None
            return data["value"]
        except (json.JSONDecodeError, KeyError, OSError):
            return None

    def set(self, key: str, value):
        path = self._key_path(key)
        with open(path, "w") as f:
            json.dump({"ts": time.time(), "value": value}, f, ensure_ascii=False)
