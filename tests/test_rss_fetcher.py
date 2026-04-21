from unittest.mock import patch, MagicMock
from fetcher.rss_fetcher import fetch_rss, _parse_feed


def _make_entry(title, link, summary="", published="2026-04-17T00:00:00Z", guid=None):
    entry = MagicMock()
    entry.get.side_effect = lambda k, d="": {
        "title": title,
        "link": link,
        "summary": summary,
        "id": guid or link,
    }.get(k, d)
    entry.title = title
    entry.link = link
    entry.get_return = None
    # feedparser uses .published_parsed or .updated_parsed
    import time
    entry.published_parsed = time.strptime("2026-04-17", "%Y-%m-%d")
    entry.get = lambda k, d=None: {
        "title": title,
        "link": link,
        "summary": summary,
        "id": guid or link,
    }.get(k, d)
    return entry


def test_parse_feed_basic():
    feed_data = MagicMock()
    feed_data.entries = [
        _make_entry(
            title="A Great Paper on Denoising",
            link="https://huggingface.co/papers/2404.11111",
            summary="We propose a denoising method.",
        ),
    ]
    papers = _parse_feed(feed_data, "HuggingFace")
    assert len(papers) == 1
    p = papers[0]
    assert "Denoising" in p.title
    assert p.source == "rss:HuggingFace"
    assert p.abstract == "We propose a denoising method."


def test_parse_feed_extracts_arxiv_id_from_url():
    feed_data = MagicMock()
    feed_data.entries = [
        _make_entry(
            title="Paper",
            link="https://huggingface.co/papers/2404.12345",
            summary="Abstract",
        ),
    ]
    papers = _parse_feed(feed_data, "HF")
    assert papers[0].id == "rss:2404.12345"


def test_parse_feed_uses_link_as_fallback_id():
    feed_data = MagicMock()
    feed_data.entries = [
        _make_entry(
            title="Paper",
            link="https://example.com/paper/123",
            summary="Abstract",
        ),
    ]
    papers = _parse_feed(feed_data, "PwC")
    assert papers[0].id == "rss:https://example.com/paper/123"


@patch("fetcher.rss_fetcher.feedparser.parse")
def test_fetch_rss_multiple_feeds(mock_parse):
    entry = _make_entry("Paper", "https://example.com/1", "Abstract")
    mock_feed = MagicMock()
    mock_feed.entries = [entry]
    mock_parse.return_value = mock_feed

    feeds = [
        {"name": "Feed1", "url": "https://example.com/rss1"},
        {"name": "Feed2", "url": "https://example.com/rss2"},
    ]
    papers = fetch_rss(feeds)
    assert mock_parse.call_count == 2
    assert len(papers) >= 1
