"""Offline parse tests for the CVPR fetcher (no network)."""
import os

from fetcher.cvpr_fetcher import (
    _parse_listing,
    _parse_abstract,
    _slug_from_path,
    _build_paper,
)

FIXTURES = os.path.join(os.path.dirname(__file__), "fixtures")


def _load(name: str) -> str:
    with open(os.path.join(FIXTURES, name)) as f:
        return f.read()


def test_parse_listing_extracts_records():
    records = _parse_listing(_load("cvpr_listing_sample.html"))
    assert len(records) >= 3
    first = records[0]
    assert first["title"] == (
        "Generalizable Structure-Aware Keypoint Correspondence "
        "for Category-Unified 3D Single Object Tracking"
    )
    assert "Jie Xiao" in first["authors"]
    assert len(first["authors"]) == 9
    assert first["detail_url"].startswith("https://openaccess.thecvf.com/content/")
    assert first["detail_url"].endswith("_paper.html")


def test_parse_listing_finds_pdf_url():
    records = _parse_listing(_load("cvpr_listing_sample.html"))
    assert records[0]["pdf_url"].endswith("_paper.pdf")
    assert "/papers/" in records[0]["pdf_url"]


def test_parse_abstract_from_detail_page():
    abstract = _parse_abstract(_load("cvpr_detail_sample.html"))
    assert abstract
    assert "3D single object tracking" in abstract


def test_slug_from_path_strips_suffix():
    path = "/content/CVPR2026/html/Xiao_Foo_Bar_CVPR_2026_paper.html"
    assert _slug_from_path(path) == "Xiao_Foo_Bar"
    pdf = "/content/CVPR2026/papers/Xiao_Foo_Bar_CVPR_2026_paper.pdf"
    assert _slug_from_path(pdf) == "Xiao_Foo_Bar"


def test_build_paper_shape():
    rec = {
        "title": "Some Title",
        "authors": ["Alice", "Bob"],
        "detail_url": "https://openaccess.thecvf.com/content/CVPR2026/html/x_paper.html",
        "pdf_url": "https://openaccess.thecvf.com/content/CVPR2026/papers/x_paper.pdf",
        "slug": "Author_Some_Title",
    }
    paper = _build_paper(rec, 2026, abstract="An abstract.")
    assert paper.id == "cvpr2026:Author_Some_Title"
    assert paper.source == "cvpr"
    assert paper.venue == "CVPR 2026"
    assert paper.categories == ["cs.CV"]
    assert paper.abstract == "An abstract."
    assert paper.published == "2026-01-01"


def test_generate_markdown_groups_by_category():
    from output.markdown_output import generate_markdown

    papers = [
        {"title": "P1", "primary_category": "图像生成", "relevance_score": 9,
         "authors": ["A"], "summary_zh": "摘要1", "url": "http://x/1"},
        {"title": "P2", "primary_category": "图像生成", "relevance_score": 7,
         "authors": ["B"], "url": "http://x/2"},
    ]
    md = generate_markdown(papers, "CVPR 2026 精选")
    assert "# CVPR 2026 精选" in md
    assert "## 图像生成（2 篇）" in md
    # higher score sorts first within the group
    assert md.index("P1") < md.index("P2")
