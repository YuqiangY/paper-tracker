from __future__ import annotations
import json
import os
from collections import defaultdict
from jinja2 import Environment, FileSystemLoader

TEMPLATE_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "templates")


def _json_or_list(val):
    if isinstance(val, str):
        try:
            return json.loads(val)
        except (json.JSONDecodeError, TypeError):
            return []
    if isinstance(val, list):
        return val
    return []


def generate_daily_page(
    papers: list[dict],
    date: str,
    output_dir: str,
    title: str,
    summary: str = "",
):
    env = Environment(loader=FileSystemLoader(TEMPLATE_DIR))
    template = env.get_template("daily.html.j2")

    categories = defaultdict(list)
    for p in papers:
        cat = p.get("primary_category", "other") or "other"
        p["tags_list"] = _json_or_list(p.get("tags"))
        p["authors_list"] = _json_or_list(p.get("authors"))
        raw_enriched = p.get("authors_enriched")
        if isinstance(raw_enriched, str):
            try:
                raw_enriched = json.loads(raw_enriched)
            except (json.JSONDecodeError, TypeError):
                raw_enriched = None
        p.setdefault("venue", None)
        p.setdefault("citation_count", None)
        p.setdefault("tldr", None)
        p["authors_enriched_list"] = raw_enriched or []
        if raw_enriched:
            h_indices = [a["h_index"] for a in raw_enriched if a.get("h_index") is not None]
            p["max_h_index"] = max(h_indices) if h_indices else None
        else:
            p["max_h_index"] = None
        categories[cat].append(p)

    html = template.render(
        title=title,
        date=date,
        papers=papers,
        categories=dict(categories),
        summary=summary,
    )

    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, f"{date}.html"), "w") as f:
        f.write(html)


def generate_index_page(
    dates: list[str],
    output_dir: str,
    title: str,
):
    env = Environment(loader=FileSystemLoader(TEMPLATE_DIR))
    template = env.get_template("index.html.j2")

    html = template.render(
        title=title,
        dates=sorted(dates, reverse=True),
    )

    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, "index.html"), "w") as f:
        f.write(html)
