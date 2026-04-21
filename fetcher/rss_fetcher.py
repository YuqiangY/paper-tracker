from __future__ import annotations
import re
import time
import feedparser
from models import Paper


def fetch_rss(feeds: list[dict]) -> list[Paper]:
    all_papers = []
    for feed_cfg in feeds:
        feed_data = feedparser.parse(feed_cfg["url"])
        papers = _parse_feed(feed_data, feed_cfg["name"])
        all_papers.extend(papers)
    return all_papers


def _parse_feed(feed_data, feed_name: str) -> list[Paper]:
    papers = []
    for entry in feed_data.entries:
        title = entry.get("title", "")
        link = entry.get("link", "")
        summary = entry.get("summary", "")

        paper_id = _extract_id(link)

        published = ""
        if hasattr(entry, "published_parsed") and entry.published_parsed:
            published = time.strftime("%Y-%m-%d", entry.published_parsed)
        elif hasattr(entry, "updated_parsed") and entry.updated_parsed:
            published = time.strftime("%Y-%m-%d", entry.updated_parsed)

        papers.append(Paper(
            id=f"rss:{paper_id}",
            title=title.strip(),
            authors=[],
            abstract=summary.strip(),
            url=link,
            source=f"rss:{feed_name}",
            published=published,
            categories=[],
        ))
    return papers


def _extract_id(url: str) -> str:
    match = re.search(r"(\d{4}\.\d{4,5})", url)
    if match:
        return match.group(1)
    return url
