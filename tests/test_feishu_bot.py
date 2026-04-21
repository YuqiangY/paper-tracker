from unittest.mock import patch, MagicMock
import json

from output.feishu_bot import send_daily_bot_message, _build_card


def test_build_card_structure():
    card = _build_card("**底层视觉**：超分方向", "2026-04-21", 42, "https://example.com/2026-04-21.html")
    assert card["header"]["title"]["content"] == "📄 Paper Tracker 今日热点 — 2026-04-21"
    assert card["header"]["template"] == "blue"
    assert len(card["elements"]) == 4  # div + div + hr + action
    assert card["elements"][0]["text"]["content"] == "**底层视觉**：超分方向"
    assert "42" in card["elements"][1]["text"]["content"]
    assert card["elements"][3]["actions"][0]["url"] == "https://example.com/2026-04-21.html"


def test_build_card_no_url():
    card = _build_card("summary", "2026-04-21", 10, "")
    assert len(card["elements"]) == 2  # no hr + action


@patch("output.feishu_bot.urllib.request.urlopen")
def test_send_success(mock_urlopen):
    mock_resp = MagicMock()
    mock_resp.read.return_value = json.dumps({"code": 0}).encode()
    mock_resp.__enter__ = MagicMock(return_value=mock_resp)
    mock_resp.__exit__ = MagicMock(return_value=False)
    mock_urlopen.return_value = mock_resp

    result = send_daily_bot_message(
        "https://hook.example.com/xxx",
        "summary text",
        "2026-04-21",
        10,
        "https://pages.example.com/2026-04-21.html",
    )
    assert result is True
    mock_urlopen.assert_called_once()


@patch("output.feishu_bot.urllib.request.urlopen")
def test_send_api_error(mock_urlopen):
    mock_resp = MagicMock()
    mock_resp.read.return_value = json.dumps({"code": 9499, "msg": "bad"}).encode()
    mock_resp.__enter__ = MagicMock(return_value=mock_resp)
    mock_resp.__exit__ = MagicMock(return_value=False)
    mock_urlopen.return_value = mock_resp

    result = send_daily_bot_message("https://hook.example.com/xxx", "text", "2026-04-21", 5, "")
    assert result is False


def test_send_no_webhook():
    result = send_daily_bot_message("", "text", "2026-04-21", 5, "")
    assert result is False
