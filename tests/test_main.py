import json
import os
import tempfile
from unittest.mock import patch, MagicMock
from main import load_config, run_pipeline
from models import Paper


def test_load_config():
    config = load_config("config.yaml")
    assert "interests" in config
    assert "sources" in config
    assert "filter" in config
    assert "output" in config
    assert len(config["interests"]) >= 1


@patch("main.fetch_arxiv")
@patch("main.fetch_rss")
@patch("main.llm_filter")
def test_run_pipeline_saves_json(mock_llm, mock_rss, mock_arxiv):
    from models import Paper

    test_paper = Paper(
        id="2404.12345",
        title="Test SR Paper",
        authors=["Alice"],
        abstract="About super-resolution denoising.",
        url="https://arxiv.org/abs/2404.12345",
        source="arxiv",
        published="2026-04-17",
        categories=["cs.CV"],
        relevance_score=8.0,
        primary_category="底层视觉",
        summary_zh="测试摘要",
        why_relevant="related",
        tags=["super-resolution"],
    )
    mock_arxiv.return_value = [test_paper]
    mock_rss.return_value = []
    mock_llm.return_value = [test_paper]

    with tempfile.TemporaryDirectory() as tmpdir:
        config = load_config("config.yaml")
        config["output"]["html"]["output_dir"] = os.path.join(tmpdir, "site")
        data_dir = os.path.join(tmpdir, "data", "daily")
        db_path = os.path.join(tmpdir, "papers.db")

        run_pipeline(config, data_dir=data_dir, db_path=db_path, site_dir=os.path.join(tmpdir, "site"))

        # Check daily JSON was saved
        json_files = [f for f in os.listdir(data_dir) if f.endswith(".json")]
        assert len(json_files) == 1

        with open(os.path.join(data_dir, json_files[0])) as f:
            papers = json.load(f)
        assert len(papers) == 1
        assert papers[0]["id"] == "2404.12345"

        # Check HTML was generated
        html_files = [f for f in os.listdir(os.path.join(tmpdir, "site")) if f.endswith(".html")]
        assert len(html_files) >= 1


@patch("main.fetch_arxiv")
@patch("main.fetch_rss")
@patch("main.llm_filter")
@patch("main.enrich_authors")
def test_run_pipeline_with_enrichment(mock_enrich, mock_llm, mock_rss, mock_arxiv):
    test_paper = Paper(
        id="2404.12345",
        title="Test SR Paper",
        authors=["Alice"],
        abstract="About super-resolution denoising.",
        url="https://arxiv.org/abs/2404.12345",
        source="arxiv",
        published="2026-04-17",
        categories=["cs.CV"],
        relevance_score=8.0,
        primary_category="底层视觉",
        summary_zh="测试摘要",
        why_relevant="related",
        tags=["super-resolution"],
        authors_enriched=[{"name": "Alice", "affiliation": "MIT", "h_index": 45, "semantic_scholar_id": "111"}],
    )
    mock_arxiv.return_value = [test_paper]
    mock_rss.return_value = []
    mock_enrich.return_value = [test_paper]
    mock_llm.return_value = [test_paper]

    with tempfile.TemporaryDirectory() as tmpdir:
        config = load_config("config.yaml")
        config["enrichment"] = {"enabled": True, "timeout": 10, "max_authors_per_paper": 5}
        config["output"]["html"]["output_dir"] = os.path.join(tmpdir, "site")
        data_dir = os.path.join(tmpdir, "data", "daily")
        db_path = os.path.join(tmpdir, "papers.db")

        run_pipeline(config, data_dir=data_dir, db_path=db_path, site_dir=os.path.join(tmpdir, "site"))

        json_files = [f for f in os.listdir(data_dir) if f.endswith(".json")]
        assert len(json_files) == 1
        with open(os.path.join(data_dir, json_files[0])) as f:
            papers = json.load(f)
        assert papers[0].get("authors_enriched") is not None
        assert papers[0]["authors_enriched"][0]["h_index"] == 45
