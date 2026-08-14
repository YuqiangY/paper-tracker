"""Offline parse tests for the AAAI fetcher (no network)."""
import os

from fetcher.aaai_fetcher import (
    _parse_archive,
    _parse_issue,
    _parse_article,
    _build_paper,
    _article_id,
    _decode,
)

FIXTURES = os.path.join(os.path.dirname(__file__), "fixtures")


def _load(name: str) -> str:
    with open(os.path.join(FIXTURES, name)) as f:
        return f.read()


def test_parse_archive_finds_aaai26_issues():
    ids = _parse_archive(_load("aaai_archive_sample.html"), 2026)
    assert len(ids) >= 5
    assert all(i.isdigit() for i in ids)
    assert "683" in ids  # AAAI-26 Technical Tracks 1


def test_parse_issue_extracts_article_urls():
    urls = _parse_issue(_load("aaai_issue_sample.html"))
    assert len(urls) >= 1
    assert all("/article/view/" in u for u in urls)
    # normalized: no /download suffix
    assert all("/download" not in u for u in urls)


def test_parse_article_fields():
    rec = _parse_article(_load("aaai_article_sample.html"))
    assert rec["title"]
    assert len(rec["authors"]) >= 1
    assert rec["pdf_url"] and rec["pdf_url"].endswith(tuple("0123456789"))
    assert rec["published"].startswith("2026")
    # abstract present and Abstract heading stripped
    assert rec["abstract"]
    assert not rec["abstract"].startswith("Abstract")


def test_build_paper_shape():
    rec = {
        "title": "Some AAAI Paper", "authors": ["A", "B"],
        "abstract": "An abstract.", "pdf_url": "https://ojs.aaai.org/x/download/1",
        "published": "2026-03-17", "doi": "10.1/x",
    }
    p = _build_paper(rec, "https://ojs.aaai.org/index.php/AAAI/article/view/36958", 2026)
    assert p.id == "aaai2026:36958"
    assert p.source == "aaai"
    assert p.venue == "AAAI 2026"
    assert p.categories == ["cs.AI"]
    assert p.published == "2026-03-17"


def test_article_id_from_url():
    assert _article_id("https://ojs.aaai.org/index.php/AAAI/article/view/36958") == "36958"
    assert _article_id("https://ojs.aaai.org/index.php/AAAI/article/view/36958/") == "36958"


def test_decode_plain_and_gzip():
    import gzip
    assert _decode(b"hello") == "hello"
    assert _decode(gzip.compress("你好".encode())) == "你好"
