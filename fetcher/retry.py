"""Shared retry utilities with exponential backoff, jitter, and Retry-After support."""
from __future__ import annotations
import random
import time
import urllib.error
import urllib.request
import json
import logging

log = logging.getLogger(__name__)


def request_with_retry(
    url: str,
    *,
    method: str = "GET",
    headers: dict[str, str] | None = None,
    data: bytes | None = None,
    timeout: int = 30,
    max_attempts: int = 4,
    base_delay: float = 5.0,
    max_delay: float = 120.0,
    retry_on: tuple[int, ...] = (429, 500, 502, 503),
) -> bytes:
    """HTTP request with exponential backoff + jitter on retryable errors.

    Returns raw response bytes. Raises on non-retryable errors or exhausted retries.
    """
    for attempt in range(max_attempts):
        try:
            req = urllib.request.Request(url, data=data, method=method)
            req.add_header("User-Agent", "PaperTracker/1.0")
            for k, v in (headers or {}).items():
                req.add_header(k, v)
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                return resp.read()
        except urllib.error.HTTPError as e:
            if e.code == 404:
                raise
            if e.code not in retry_on or attempt == max_attempts - 1:
                raise
            delay = _calc_delay(e, attempt, base_delay, max_delay)
            log.debug("HTTP %d on %s, retry %d/%d in %.1fs", e.code, url[:80], attempt + 1, max_attempts, delay)
            time.sleep(delay)
        except (TimeoutError, OSError) as e:
            if attempt == max_attempts - 1:
                raise
            delay = _calc_delay(None, attempt, base_delay, max_delay)
            log.debug("Timeout/OS error on %s, retry %d/%d in %.1fs", url[:80], attempt + 1, max_attempts, delay)
            time.sleep(delay)


def _calc_delay(
    error: urllib.error.HTTPError | None,
    attempt: int,
    base_delay: float,
    max_delay: float,
) -> float:
    """Exponential backoff with jitter. Respects Retry-After header if present."""
    if error is not None:
        retry_after = error.headers.get("Retry-After") if hasattr(error, "headers") else None
        if retry_after:
            try:
                return min(float(retry_after) + random.uniform(0.5, 2.0), max_delay)
            except (ValueError, TypeError):
                pass

    delay = base_delay * (2 ** attempt)
    jitter = random.uniform(0, delay * 0.3)
    return min(delay + jitter, max_delay)
