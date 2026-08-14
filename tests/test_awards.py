"""Offline tests for award fetchers and pin-to-top rendering."""
import json
import os

from fetcher.papercopilot_fetcher import _parse, _build_paper
from fetcher.best_paper_md_fetcher import _parse_section

FIXTURES = os.path.join(os.path.dirname(__file__), "fixtures")


def _load(name: str) -> str:
    with open(os.path.join(FIXTURES, name)) as f:
        return f.read()


# --- Paper Copilot JSON parsing -------------------------------------------

def test_papercopilot_filters_by_status():
    data = json.loads(_load("papercopilot_cvpr_sample.json"))
    awards = _parse(data, ["Award Candidate"], "cvpr", 2025)
    assert len(awards) >= 1
    assert all(p.primary_category == "🏆 Award Candidate" for p in awards)
    # posters excluded
    assert all(p.source == "award" for p in awards)


def test_papercopilot_widen_statuses():
    data = json.loads(_load("papercopilot_cvpr_sample.json"))
    both = _parse(data, ["Award Candidate", "Highlight"], "cvpr", 2025)
    cats = {p.primary_category for p in both}
    assert "🏆 Award Candidate" in cats
    assert "🏆 Highlight" in cats


def test_papercopilot_field_mapping():
    item = {
        "title": "Test Paper", "status": "Award Candidate",
        "author": "Alice Smith; Bob Jones", "abstract": "An abstract.",
        "arxiv": "2503.10148", "pdf": "https://x/p.pdf",
        "oa": "https://openaccess/x.html",
    }
    p = _build_paper(item, "Award Candidate", "cvpr", 2025)
    assert p.id == "cvpr2025:pc:2503.10148"
    assert p.authors == ["Alice Smith", "Bob Jones"]
    assert p.abstract == "An abstract."
    assert p.url == "https://openaccess/x.html"  # oa wins
    assert p.primary_category == "🏆 Award Candidate"


def test_papercopilot_url_fallback_to_arxiv():
    item = {"title": "T", "status": "Best Paper", "author": "A",
            "arxiv": "2503.10148"}
    p = _build_paper(item, "Best Paper", "cvpr", 2026)
    assert p.url == "https://arxiv.org/abs/2503.10148"


# --- Markdown fallback parsing --------------------------------------------

def test_markdown_parses_cvpr_awards():
    md = _load("bestpapers_sample.md")
    papers = _parse_section(md, "cvpr", 2026)
    assert len(papers) == 2
    assert any("Best Paper" in p.primary_category for p in papers)
    assert all("arxiv.org" in p.url for p in papers)  # both have arxiv links
    assert all(len(p.authors) > 1 for p in papers)


def test_markdown_parses_aaai_awards():
    md = _load("bestpapers_sample.md")
    papers = _parse_section(md, "aaai", 2026)
    assert len(papers) == 7  # 5 outstanding + 2 AISI
    assert all(p.source == "award" for p in papers)
    assert all(p.authors for p in papers)


def test_markdown_missing_conf_returns_empty():
    md = _load("bestpapers_sample.md")
    assert _parse_section(md, "nonexistent", 2026) == []


# --- Pin-to-top rendering --------------------------------------------------

def test_markdown_output_pins_awards_first():
    from output.markdown_output import generate_markdown
    papers = [
        {"title": "Interest paper", "primary_category": "底层视觉", "relevance_score": 9,
         "authors": ["A"], "url": "http://x/1"},
        {"title": "Award paper", "primary_category": "🏆 Best Paper",
         "authors": ["B"], "url": "http://x/2"},
    ]
    md = generate_markdown(papers, "Test")
    assert md.index("🏆 Best Paper") < md.index("底层视觉")


def test_html_output_pins_awards_first(tmp_path):
    from output.html_output import generate_daily_page
    papers = [
        {"title": "Interest paper", "primary_category": "底层视觉", "relevance_score": 9,
         "authors": ["A"], "tags": [], "summary_zh": "x", "url": "http://x/1"},
        {"title": "Award paper", "primary_category": "🏆 Best Paper", "relevance_score": None,
         "authors": ["B"], "tags": [], "summary_zh": "y", "url": "http://x/2"},
    ]
    generate_daily_page(papers, "test", str(tmp_path), "T")
    html = (tmp_path / "test.html").read_text()
    assert html.index("🏆 Best Paper") < html.index("底层视觉")
