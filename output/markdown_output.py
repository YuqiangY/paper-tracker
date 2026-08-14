"""Render filtered papers to a Markdown digest, grouped by primary category."""
from __future__ import annotations
import json
from collections import defaultdict


def _as_list(val) -> list:
    if isinstance(val, list):
        return val
    if isinstance(val, str):
        try:
            parsed = json.loads(val)
            return parsed if isinstance(parsed, list) else []
        except (json.JSONDecodeError, TypeError):
            return []
    return []


def generate_markdown(papers: list[dict], title: str) -> str:
    """Build a Markdown document from filtered paper dicts.

    Papers are grouped by ``primary_category`` and sorted by
    ``relevance_score`` (desc) within each group.
    """
    lines: list[str] = [f"# {title}", ""]
    lines.append(f"共 {len(papers)} 篇相关论文。")
    lines.append("")

    grouped: dict[str, list[dict]] = defaultdict(list)
    for p in papers:
        grouped[p.get("primary_category") or "其他"].append(p)

    # Award categories (🏆-prefixed) first, then the rest alphabetically.
    def _cat_key(c: str) -> tuple:
        return (0 if c.startswith("🏆") else 1, c)

    for category in sorted(grouped, key=_cat_key):
        group = sorted(
            grouped[category],
            key=lambda p: p.get("relevance_score") or 0,
            reverse=True,
        )
        lines.append(f"## {category}（{len(group)} 篇）")
        lines.append("")
        for p in group:
            lines.extend(_render_paper(p))

    return "\n".join(lines)


def _render_paper(p: dict) -> list[str]:
    score = p.get("relevance_score")
    score_str = f"（相关度 {score:g}）" if score is not None else ""
    authors = _as_list(p.get("authors"))
    tags = _as_list(p.get("tags"))

    out = [f"### {p.get('title', '(无标题)')} {score_str}".rstrip(), ""]
    if authors:
        out.append(f"**作者**：{', '.join(authors)}")
    if p.get("summary_zh"):
        out.append(f"**摘要**：{p['summary_zh']}")
    if p.get("why_relevant"):
        out.append(f"**相关性**：{p['why_relevant']}")
    if tags:
        out.append(f"**标签**：{', '.join(tags)}")

    links = []
    if p.get("url"):
        links.append(f"[详情]({p['url']})")
    if p.get("pdf_url"):
        links.append(f"[PDF]({p['pdf_url']})")
    if links:
        out.append(" · ".join(links))

    out.append("")
    return out
