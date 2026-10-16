import json
import os
import tempfile
from output.html_output import generate_daily_page, generate_index_page


SAMPLE_PAPERS = [
    {
        "id": "2404.12345",
        "title": "Super-Resolution Method",
        "authors": '["Alice", "Bob"]',
        "abstract": "About SR.",
        "url": "https://arxiv.org/abs/2404.12345",
        "source": "arxiv",
        "published": "2026-04-17",
        "categories": '["cs.CV"]',
        "relevance_score": 85.0,
        "primary_category": "底层视觉",
        "summary_zh": "一种新的超分方法",
        "tags": '["super-resolution"]',
    },
    {
        "id": "2404.67890",
        "title": "Video Generation Model",
        "authors": '["Charlie"]',
        "abstract": "About video.",
        "url": "https://arxiv.org/abs/2404.67890",
        "source": "arxiv",
        "published": "2026-04-17",
        "categories": '["cs.CV"]',
        "relevance_score": 70.0,
        "primary_category": "视频算法",
        "summary_zh": "视频生成模型",
        "tags": '["video generation"]',
    },
]


def test_generate_daily_page():
    with tempfile.TemporaryDirectory() as tmpdir:
        generate_daily_page(
            papers=SAMPLE_PAPERS,
            date="2026-04-17",
            output_dir=tmpdir,
            title="Paper Tracker",
        )
        path = os.path.join(tmpdir, "2026-04-17.html")
        assert os.path.exists(path)
        content = open(path).read()
        assert "Super-Resolution Method" in content
        assert "Video Generation Model" in content
        assert "底层视觉" in content
        assert "视频算法" in content
        assert ">85<" in content


def test_generate_index_page():
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create a daily page first so index can list it
        generate_daily_page(SAMPLE_PAPERS, "2026-04-17", tmpdir, "Test")
        generate_index_page(
            dates=["2026-04-17"],
            output_dir=tmpdir,
            title="Paper Tracker",
        )
        path = os.path.join(tmpdir, "index.html")
        assert os.path.exists(path)
        content = open(path).read()
        assert "2026-04-17" in content
        assert "Paper Tracker" in content


def test_generate_index_page_with_features():
    with tempfile.TemporaryDirectory() as tmpdir:
        generate_index_page(
            dates=["2026-04-17", "cvpr2026"],  # cvpr2026 is not a date
            output_dir=tmpdir,
            title="Paper Tracker",
            features=[{"slug": "cvpr2026", "title": "CVPR 2026 精选"}],
        )
        content = open(os.path.join(tmpdir, "index.html")).read()
        # feature title + link present
        assert "CVPR 2026 精选" in content
        assert 'href="cvpr2026.html"' in content
        # non-date slug is NOT listed as a daily date entry
        assert 'href="2026-04-17.html"' in content
        assert content.count("cvpr2026.html") == 1  # only in features, not dates


def test_daily_page_groups_by_category():
    with tempfile.TemporaryDirectory() as tmpdir:
        generate_daily_page(SAMPLE_PAPERS, "2026-04-17", tmpdir, "Test")
        content = open(os.path.join(tmpdir, "2026-04-17.html")).read()
        assert "底层视觉" in content
        assert "视频算法" in content


def test_daily_page_shows_enriched_authors():
    papers = [{
        **SAMPLE_PAPERS[0],
        "authors_enriched": json.dumps([
            {"name": "Alice", "affiliation": "MIT", "h_index": 45, "semantic_scholar_id": None},
            {"name": "Bob", "affiliation": "Stanford", "h_index": 30, "semantic_scholar_id": None},
        ]),
    }]
    with tempfile.TemporaryDirectory() as tmpdir:
        generate_daily_page(papers, "2026-04-17", tmpdir, "Test")
        content = open(os.path.join(tmpdir, "2026-04-17.html")).read()
        assert "MIT" in content
        assert "h=45" in content
        assert "Stanford" in content
        assert "max h=45" in content


def test_daily_page_handles_no_enrichment():
    with tempfile.TemporaryDirectory() as tmpdir:
        generate_daily_page(SAMPLE_PAPERS, "2026-04-17", tmpdir, "Test")
        content = open(os.path.join(tmpdir, "2026-04-17.html")).read()
        assert "Alice" in content


def test_daily_page_shows_integer_score_and_subscores():
    papers = [{"title": "T", "url": "u", "relevance_score": 85.0, "score_relevance": 36,
               "score_novelty": 24, "score_quality": 25, "primary_category": "底层视觉",
               "summary_zh": "s", "tags": "[]", "authors": '["A"]'}]
    with tempfile.TemporaryDirectory() as tmpdir:
        generate_daily_page(papers, "2026-10-15", tmpdir, "PT")
        with open(os.path.join(tmpdir, "2026-10-15.html")) as f:
            html = f.read()
    assert ">85<" in html
    assert "相关 36 · 新颖 24 · 质量 25" in html


def test_daily_page_renders_legacy_paper_without_subscores():
    papers = [{"title": "T", "url": "u", "relevance_score": 80.0, "primary_category": "底层视觉",
               "summary_zh": "s", "tags": "[]", "authors": '["A"]'}]
    with tempfile.TemporaryDirectory() as tmpdir:
        generate_daily_page(papers, "2026-10-15", tmpdir, "PT")
        with open(os.path.join(tmpdir, "2026-10-15.html")) as f:
            html = f.read()
    assert ">80<" in html and "新颖" not in html
