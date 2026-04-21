from __future__ import annotations
import json
import logging
import urllib.request
import urllib.error

log = logging.getLogger(__name__)


def send_daily_bot_message(
    webhook_url: str,
    summary_text: str,
    date: str,
    paper_count: int,
    pages_url: str = "",
) -> bool:
    if not webhook_url:
        log.warning("Feishu bot webhook URL not configured")
        return False

    card = _build_card(summary_text, date, paper_count, pages_url)
    payload = json.dumps({"msg_type": "interactive", "card": card}).encode()

    req = urllib.request.Request(
        webhook_url,
        data=payload,
        headers={"Content-Type": "application/json"},
    )

    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            body = json.loads(resp.read())
            if body.get("code") == 0:
                log.info("Feishu bot message sent successfully")
                return True
            else:
                log.warning("Feishu bot API error: %s", body)
                return False
    except urllib.error.URLError as e:
        log.warning("Feishu bot request failed: %s", e)
        return False
    except Exception as e:
        log.warning("Feishu bot unexpected error: %s", e)
        return False


def _build_card(summary_text: str, date: str, paper_count: int, pages_url: str) -> dict:
    elements = [
        {
            "tag": "div",
            "text": {
                "tag": "lark_md",
                "content": summary_text,
            },
        },
        {
            "tag": "div",
            "text": {
                "tag": "lark_md",
                "content": f"共筛选 **{paper_count}** 篇高相关论文",
            },
        },
    ]

    if pages_url:
        elements.append({"tag": "hr"})
        elements.append({
            "tag": "action",
            "actions": [
                {
                    "tag": "button",
                    "text": {"tag": "plain_text", "content": "查看完整报告"},
                    "url": pages_url,
                    "type": "primary",
                },
            ],
        })

    return {
        "header": {
            "title": {"tag": "plain_text", "content": f"📄 Paper Tracker 今日热点 — {date}"},
            "template": "blue",
        },
        "elements": elements,
    }
