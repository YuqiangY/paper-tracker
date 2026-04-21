from __future__ import annotations
import json
import logging
import markdown
from collections import defaultdict

import anthropic

log = logging.getLogger(__name__)


def generate_summary(
    papers: list[dict],
    interests: list[dict],
    model: str,
    api_key: str,
    base_url: str | None = None,
) -> tuple[str, str]:
    """Returns (html, plain_text) tuple. Both empty on failure."""
    if not papers:
        return "", ""

    prompt = _build_prompt(papers, interests)

    try:
        raw_text = _call_llm(prompt, model, api_key, base_url)
    except Exception as e:
        log.warning("Summary generation failed: %s", e)
        return "", ""

    if not raw_text.strip():
        return "", ""

    normalized = "\n\n".join(line for line in raw_text.strip().splitlines() if line.strip())
    html = markdown.markdown(normalized, extensions=["tables", "fenced_code"])
    return html, normalized


def _build_prompt(papers: list[dict], interests: list[dict]) -> str:
    grouped: dict[str, list[dict]] = defaultdict(list)
    for p in papers:
        cat = p.get("primary_category") or "其他"
        grouped[cat].append(p)

    sections = []
    for cat, cat_papers in grouped.items():
        lines = [f"## {cat} ({len(cat_papers)} 篇)"]
        for p in sorted(cat_papers, key=lambda x: x.get("relevance_score") or 0, reverse=True):
            title = p.get("title", "")
            summary = p.get("summary_zh", "")
            tags = _parse_json(p.get("tags"))
            score = p.get("relevance_score", "")
            lines.append(f"- [{score}分] {title}")
            if summary:
                lines.append(f"  摘要: {summary}")
            if tags:
                lines.append(f"  标签: {', '.join(tags)}")
        sections.append("\n".join(lines))

    interest_desc = "\n".join(
        f"- {i['name']}: {i.get('description', '')}" for i in interests
    )

    papers_text = "\n\n".join(sections)

    return f"""你是一位 AI 研究领域分析师。以下是今日筛选出的高相关度论文，按研究方向分组。

请对每个方向进行极简热点总结，严格要求：
- 每个方向限 100 字以内（1-2句话）
- 提炼核心趋势关键词，不要展开解释
- 无相关论文的方向直接跳过，不要输出
- 最后附一句跨领域共同趋势（如有，同样 100 字以内）

输出格式：
- 每个方向单独一段（用空行分隔），格式为：**方向名**：总结内容
- 跨领域趋势格式为：**跨领域趋势**：总结内容
- 语言：中文
- 不需要 Markdown 标题

我关注的研究方向：
{interest_desc}

今日论文：

{papers_text}"""


def _call_llm(prompt: str, model: str, api_key: str, base_url: str | None) -> str:
    kwargs = {"api_key": api_key}
    if base_url:
        kwargs["base_url"] = base_url
    client = anthropic.Anthropic(**kwargs)
    message = client.messages.create(
        model=model,
        max_tokens=2048,
        messages=[{"role": "user", "content": prompt}],
    )
    for block in message.content:
        if block.type == "text":
            return block.text
    return ""


def _parse_json(value) -> list:
    if isinstance(value, list):
        return value
    if isinstance(value, str):
        try:
            return json.loads(value)
        except (json.JSONDecodeError, TypeError):
            return []
    return []
