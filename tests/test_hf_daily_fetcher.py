from unittest.mock import patch, MagicMock
from fetcher.hf_daily_fetcher import fetch_hf_daily, _parse_hf_response

SAMPLE_HF_RESPONSE = [
    {
        "paper": {
            "id": "2604.16272",
            "title": "A Great Vision Model",
            "summary": "We propose a new vision model that achieves SOTA.",
            "authors": [
                {"_id": "a1", "name": "Alice", "hidden": False},
                {"_id": "a2", "name": "Bob", "hidden": False},
            ],
            "publishedAt": "2026-04-18T12:00:00.000Z",
            "upvotes": 42,
            "ai_keywords": ["vision", "transformer"],
        },
        "publishedAt": "2026-04-18T20:00:00.000Z",
        "title": "A Great Vision Model",
        "numComments": 3,
    },
    {
        "paper": {
            "id": "2604.15000",
            "title": "Old Paper",
            "summary": "This is an older paper.",
            "authors": [{"_id": "a3", "name": "Charlie", "hidden": False}],
            "publishedAt": "2026-04-10T12:00:00.000Z",
            "upvotes": 5,
        },
        "publishedAt": "2026-04-10T20:00:00.000Z",
        "title": "Old Paper",
        "numComments": 0,
    },
    {
        "paper": {
            "id": "2604.16500",
            "title": "Another Recent Paper",
            "summary": "Another recent contribution.",
            "authors": [{"_id": "a4", "name": "Dave", "hidden": False}],
            "publishedAt": "2026-04-19T08:00:00.000Z",
            "upvotes": 10,
            "ai_keywords": ["diffusion"],
        },
        "publishedAt": "2026-04-19T08:00:00.000Z",
        "title": "Another Recent Paper",
        "numComments": 1,
    },
]


def test_parse_hf_response_basic():
    papers = _parse_hf_response(SAMPLE_HF_RESPONSE, "2026-04-15")
    assert len(papers) == 2
    p = papers[0]
    assert p.id == "2604.16272"
    assert p.title == "A Great Vision Model"
    assert p.authors == ["Alice", "Bob"]
    assert "SOTA" in p.abstract
    assert p.url == "https://huggingface.co/papers/2604.16272"
    assert p.source == "hf_daily"
    assert p.published == "2026-04-18"
    assert p.pdf_url == "https://arxiv.org/pdf/2604.16272"
    assert p.categories == ["vision", "transformer"]


def test_parse_hf_response_date_filter():
    papers = _parse_hf_response(SAMPLE_HF_RESPONSE, "2026-04-18")
    ids = [p.id for p in papers]
    assert "2604.16272" in ids
    assert "2604.16500" in ids
    assert "2604.15000" not in ids


def test_parse_hf_response_strict_cutoff():
    papers = _parse_hf_response(SAMPLE_HF_RESPONSE, "2026-04-20")
    assert len(papers) == 0


def test_parse_hf_response_dedup():
    duped = SAMPLE_HF_RESPONSE + [SAMPLE_HF_RESPONSE[0]]
    papers = _parse_hf_response(duped, "2026-04-01")
    ids = [p.id for p in papers]
    assert ids.count("2604.16272") == 1


def test_parse_hf_response_missing_paper_field():
    bad_entries = [{"title": "no paper field"}]
    papers = _parse_hf_response(bad_entries, "2026-04-01")
    assert len(papers) == 0


def test_parse_hf_response_no_ai_keywords():
    papers = _parse_hf_response(SAMPLE_HF_RESPONSE, "2026-04-01")
    old_paper = [p for p in papers if p.id == "2604.15000"][0]
    assert old_paper.categories == []


@patch("fetcher.hf_daily_fetcher._query_hf_api")
def test_fetch_hf_daily_integration(mock_api):
    mock_api.return_value = SAMPLE_HF_RESPONSE
    papers = fetch_hf_daily(lookback_days=30)
    assert len(papers) == 3
    mock_api.assert_called_once()


@patch("fetcher.hf_daily_fetcher._query_hf_api")
def test_fetch_hf_daily_api_failure(mock_api):
    mock_api.return_value = []
    papers = fetch_hf_daily(lookback_days=3)
    assert papers == []
