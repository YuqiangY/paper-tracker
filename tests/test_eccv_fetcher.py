"""Offline parse tests for the ECCV fetcher (no network)."""
import os

from fetcher.eccv_fetcher import _parse_markdown, _build_paper, _clean_category, _slug

FIXTURES = os.path.join(os.path.dirname(__file__), "fixtures")


def _load(name: str) -> str:
    with open(os.path.join(FIXTURES, name)) as f:
        return f.read()


def test_parse_markdown_extracts_papers():
    records = _parse_markdown(_load("eccv2026_sample.md"))
    # 85 titles in fixture, minus the single "xxx" placeholder
    assert len(records) >= 50
    titles = [r["title"] for r in records]
    assert "xxx" not in titles
    # a known paper with arxiv link
    allo = next((r for r in records if r["title"].startswith("Allo")), None)
    assert allo is not None
    assert allo["arxiv_id"] == "2604.19238"
    assert allo["category"]  # section category assigned


def test_parse_markdown_assigns_categories():
    records = _parse_markdown(_load("eccv2026_sample.md"))
    cats = {r["category"] for r in records}
    # sub-task sections like 超分辨率 present
    assert any("超分辨率" in c for c in cats)


def test_clean_category_strips_english_paren():
    assert _clean_category("超分辨率(Super-Resolution)") == "超分辨率"
    assert _clean_category("去噪（Denoising）") == "去噪"


def test_build_paper_with_arxiv():
    from models import Paper
    rec = {"title": "T", "arxiv_id": "2604.19238", "category": "超分辨率"}
    arxiv_paper = Paper(
        id="2604.19238", title="T", authors=["X"], abstract="abs",
        url="https://arxiv.org/abs/2604.19238", source="arxiv",
        published="2026-04-01", categories=["cs.CV"],
        pdf_url="https://arxiv.org/pdf/2604.19238",
    )
    p = _build_paper(rec, 2026, arxiv_paper)
    assert p.id == "eccv2026:2604.19238"
    assert p.source == "eccv"
    assert p.abstract == "abs"
    assert p.authors == ["X"]
    assert p.venue == "ECCV 2026"


def test_build_paper_without_arxiv():
    rec = {"title": "No Arxiv Paper", "arxiv_id": None, "code_url": None, "category": "去噪"}
    p = _build_paper(rec, 2026, None)
    assert p.id.startswith("eccv2026:no-arxiv-paper")
    assert p.abstract == ""
    assert p.authors == []
    assert p.url == ""


def test_build_paper_falls_back_to_code_url():
    rec = {"title": "Code Only Paper", "arxiv_id": None,
           "code_url": "https://github.com/foo/bar", "category": "去噪"}
    p = _build_paper(rec, 2026, None)
    assert p.url == "https://github.com/foo/bar"


def test_parse_markdown_captures_code_url():
    records = _parse_markdown(_load("eccv2026_sample.md"))
    # AVSR-Diff has both a Paper (arxiv) and a Code link in the fixture
    avsr = next((r for r in records if r["title"].startswith("AVSR-Diff")), None)
    assert avsr is not None
    assert avsr["code_url"] and "github.com" in avsr["code_url"]
