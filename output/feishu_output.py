from __future__ import annotations
import json
import logging
import subprocess
import tempfile
from collections import Counter

log = logging.getLogger(__name__)


def _parse_json_field(value):
    if isinstance(value, str):
        try:
            return json.loads(value)
        except (json.JSONDecodeError, TypeError):
            return None
    return value


def _format_author(paper: dict, rich: bool = False) -> str:
    enriched = _parse_json_field(paper.get("authors_enriched"))
    if enriched:
        parts = []
        for a in enriched[:3]:
            name = a.get("name", "")
            aff = a.get("affiliation")
            h = a.get("h_index")
            if rich:
                s = f"**{name}**"
                if aff:
                    s += f" *({aff})*"
                if h:
                    s += f" `h={h}`"
            else:
                s = name
                if aff:
                    s += f" ({aff})"
                if h:
                    s += f" h={h}"
            parts.append(s)
        total = len(enriched)
        if total > 3:
            parts.append(f"等共 {total} 人")
        return ", ".join(parts)

    authors = _parse_json_field(paper.get("authors"))
    if authors and isinstance(authors, list):
        if len(authors) <= 2:
            return ", ".join(authors)
        return f"{authors[0]}, {authors[1]} 等"
    return ""


def _format_tags(paper: dict, rich: bool = False) -> str:
    tags = _parse_json_field(paper.get("tags"))
    if tags and isinstance(tags, list):
        if rich:
            return " ".join(f"`{t}`" for t in tags)
        return ", ".join(tags)
    return ""


def _max_h_index(paper: dict) -> int | None:
    enriched = _parse_json_field(paper.get("authors_enriched"))
    if not enriched:
        return None
    indices = [a.get("h_index") for a in enriched if a.get("h_index")]
    return max(indices) if indices else None


def _score_emoji(score) -> str:
    if score is None or score == "?":
        return ""
    s = float(score)
    if s >= 9:
        return "fire"
    if s >= 8:
        return "star"
    return "page_facing_up"


def _build_markdown(papers: list[dict], date: str) -> str:
    cats = Counter(p.get("primary_category", "其他") for p in papers)
    cat_summary = "、".join(f"{c} {n} 篇" for c, n in cats.most_common())

    lines = [
        '<callout emoji="fire" background-color="light-blue">',
        "",
        f"**今日概览** — 共收录 **{len(papers)}** 篇论文：{cat_summary}",
        "",
        "</callout>",
        "",
        "---",
        "",
        '## 今日推荐 TOP 3 {color="blue"}',
        "",
    ]

    sorted_papers = sorted(papers, key=lambda p: p.get("relevance_score") or 0, reverse=True)
    top3 = sorted_papers[:3]
    callout_colors = ["light-yellow", "light-green", "light-green"]
    callout_emojis = ["trophy", "star", "star"]

    for i, p in enumerate(top3):
        title = p.get("title", "Untitled")
        score = p.get("relevance_score", "?")
        author_str = _format_author(p, rich=True)
        summary = p.get("summary_zh", "")
        why = p.get("why_relevant", "")
        url = p.get("url", "")
        tags = _format_tags(p, rich=True)
        max_h = _max_h_index(p)

        lines.extend([
            f'<callout emoji="{callout_emojis[i]}" background-color="{callout_colors[i]}">',
            "",
            f"### {i + 1}. [{title}]({url}) — 评分 {score}" if url else f"### {i + 1}. {title} — 评分 {score}",
            "",
        ])
        lines.append(f"**作者**: {author_str}")
        if max_h:
            lines[-1] += f"  |  max h-index: **{max_h}**"
        lines.append("")
        venue = (p.get("venue") or "").strip()
        citations = p.get("citation_count")
        if venue or (citations and citations > 0):
            meta_parts = []
            if venue:
                meta_parts.append(f"**会议/期刊**: {venue}")
            if citations and citations > 0:
                meta_parts.append(f"**引用**: {citations}")
            lines.append("  |  ".join(meta_parts))
            lines.append("")
        lines.append(f"> {summary}")
        lines.append("")
        if why:
            lines.append(f"**推荐理由**: {why}")
            lines.append("")
        if tags:
            lines.append(tags)
            lines.append("")
        lines.extend([
            "</callout>",
            "",
        ])

    grouped: dict[str, list[dict]] = {}
    for p in sorted_papers:
        cat = p.get("primary_category", "其他")
        grouped.setdefault(cat, []).append(p)

    for cat, cat_papers in grouped.items():
        lines.extend([
            "---",
            "",
            f'## {cat} ({len(cat_papers)} 篇) {{color="purple"}}',
            "",
        ])
        for p in cat_papers:
            title = p.get("title", "Untitled")
            url = p.get("url", "")
            score = p.get("relevance_score", "?")
            emoji = _score_emoji(score)
            author_str = _format_author(p, rich=True)
            summary = p.get("summary_zh") or ""
            why = p.get("why_relevant", "")
            tags = _format_tags(p, rich=True)
            max_h = _max_h_index(p)

            title_md = f"[{title}]({url})" if url else title
            bg = "light-yellow" if float(score or 0) >= 8 else "light-grey"
            emoji_attr = f' emoji="{emoji}"' if emoji else ""

            lines.extend([
                f'<callout{emoji_attr} background-color="{bg}">',
                "",
                f"**{title_md}** — 评分 **{score}**",
                "",
                f"**作者**: {author_str}",
            ])
            if max_h:
                lines[-1] += f"  |  max h-index: **{max_h}**"
            lines.append("")
            venue = (p.get("venue") or "").strip()
            citations = p.get("citation_count")
            if venue or (citations and citations > 0):
                meta_parts = []
                if venue:
                    meta_parts.append(f"**会议/期刊**: {venue}")
                if citations and citations > 0:
                    meta_parts.append(f"**引用**: {citations}")
                lines.append("  |  ".join(meta_parts))
                lines.append("")
            lines.append(f"> {summary}")
            lines.append("")
            if why:
                lines.append(f"**推荐理由**: {why}")
                lines.append("")
            if tags:
                lines.append(tags)
                lines.append("")
            lines.extend([
                "</callout>",
                "",
            ])

    return "\n".join(lines)


def generate_feishu_doc(
    papers: list[dict],
    date: str,
    wiki_space: str = "",
):
    if not papers:
        log.info("No papers to publish to Feishu")
        return

    title = f"论文追踪 {date}"
    markdown = _build_markdown(papers, date)

    tmp_file = tempfile.NamedTemporaryFile(mode="w", suffix=".md", delete=False)
    tmp_file.write(markdown)
    tmp_file.close()

    cmd = ["feishu", "docx", "create", title, "-f", tmp_file.name]
    if wiki_space:
        cmd.extend(["--wiki-space", wiki_space])

    log.info("Creating Feishu doc for %s via feishu CLI", date)
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=120,
        )
        if result.returncode == 0:
            log.info("Feishu doc created successfully")
            if result.stdout.strip():
                log.info("Output: %s", result.stdout.strip()[:500])
        elif result.returncode == 2:
            log.warning("Feishu doc created but content write failed: %s", result.stdout.strip()[:500])
        else:
            detail = result.stderr.strip() or result.stdout.strip()
            log.warning("Feishu doc creation failed (rc=%d): %s", result.returncode, detail[:500])
    except FileNotFoundError:
        log.warning("'feishu' CLI not found. Install via: npm install -g @mi/feishu@latest")
    except subprocess.TimeoutExpired:
        log.warning("Feishu doc creation timed out after 120s")
    except Exception as e:
        log.warning("Feishu doc creation failed: %s", e)
    finally:
        import os
        os.unlink(tmp_file.name)
