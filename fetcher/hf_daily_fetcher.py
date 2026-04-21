from __future__ import annotations
import json
import logging
import time
import urllib.error
import urllib.request
from datetime import datetime, timedelta, timezone
from models import Paper

log = logging.getLogger(__name__)

HF_DAILY_API = "https://huggingface.co/api/daily_papers"


def fetch_hf_daily(lookback_days: int = 3) -> list[Paper]:
    data = _query_hf_api()
    if not data:
        return []
    cutoff = (datetime.now(timezone.utc) - timedelta(days=lookback_days)).strftime("%Y-%m-%d")
    return _parse_hf_response(data, cutoff)


def _query_hf_api() -> list[dict]:
    for attempt in range(3):
        try:
            req = urllib.request.Request(HF_DAILY_API)
            req.add_header("User-Agent", "PaperTracker/1.0")
            with urllib.request.urlopen(req, timeout=30) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except (urllib.error.HTTPError, TimeoutError, OSError, json.JSONDecodeError) as e:
            log.warning("HF Daily Papers API attempt %d failed: %s", attempt + 1, e)
            if attempt < 2:
                time.sleep(5 * (attempt + 1))
                continue
            log.error("HF Daily Papers API failed after 3 attempts")
            return []
    return []


def _parse_hf_response(entries: list[dict], cutoff: str) -> list[Paper]:
    seen: dict[str, Paper] = {}

    for entry in entries:
        paper_data = entry.get("paper")
        if not paper_data:
            continue

        paper_id = paper_data.get("id", "")
        if not paper_id or paper_id in seen:
            continue

        published = entry.get("publishedAt", "")[:10]
        if published < cutoff:
            continue

        title = paper_data.get("title", "").strip()
        abstract = paper_data.get("summary", "").strip()
        authors = [a.get("name", "") for a in paper_data.get("authors", []) if a.get("name")]
        ai_keywords = paper_data.get("ai_keywords") or []

        seen[paper_id] = Paper(
            id=paper_id,
            title=title,
            authors=authors,
            abstract=abstract,
            url=f"https://huggingface.co/papers/{paper_id}",
            source="hf_daily",
            published=published,
            categories=ai_keywords,
            pdf_url=f"https://arxiv.org/pdf/{paper_id}",
        )

    log.info("HF Daily Papers: %d papers after date filter (cutoff=%s)", len(seen), cutoff)
    return list(seen.values())
