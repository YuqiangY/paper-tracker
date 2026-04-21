from unittest.mock import patch
from fetcher.s2_search_fetcher import fetch_s2_search, _parse_s2_papers

SAMPLE_S2_RESPONSE = {
    "total": 100,
    "offset": 0,
    "data": [
        {
            "paperId": "abc123",
            "title": "Image Super-Resolution via Transformers",
            "abstract": "We propose a transformer-based approach to SISR.",
            "authors": [
                {"authorId": "1001", "name": "Alice Smith"},
                {"authorId": "1002", "name": "Bob Jones"},
            ],
            "year": 2026,
            "externalIds": {"ArXiv": "2604.12345", "DOI": "10.1234/foo"},
            "publicationDate": "2026-04-15",
        },
        {
            "paperId": "def456",
            "title": "Video Diffusion Models",
            "abstract": "A diffusion-based video generation method.",
            "authors": [
                {"authorId": "2001", "name": "Charlie Brown"},
            ],
            "year": 2026,
            "externalIds": {"ArXiv": "2604.67890"},
            "publicationDate": "2026-04-16",
        },
        {
            "paperId": "ghi789",
            "title": "Non-arXiv Paper on Vision",
            "abstract": "Published only on S2.",
            "authors": [{"authorId": "3001", "name": "Dave Wilson"}],
            "year": 2026,
            "externalIds": {"DOI": "10.5678/bar"},
            "publicationDate": "2026-03-20",
        },
    ],
}


def test_parse_s2_papers_basic():
    papers = _parse_s2_papers(SAMPLE_S2_RESPONSE["data"])
    assert len(papers) == 3

    p = papers[0]
    assert p.id == "2604.12345"
    assert p.title == "Image Super-Resolution via Transformers"
    assert p.authors == ["Alice Smith", "Bob Jones"]
    assert "SISR" in p.abstract
    assert p.url == "https://arxiv.org/abs/2604.12345"
    assert p.pdf_url == "https://arxiv.org/pdf/2604.12345"
    assert p.source == "s2_search"
    assert p.published == "2026-04-15"


def test_parse_s2_papers_arxiv_id_extraction():
    papers = _parse_s2_papers(SAMPLE_S2_RESPONSE["data"])
    assert papers[0].id == "2604.12345"
    assert papers[1].id == "2604.67890"
    assert papers[2].id == "s2:ghi789"


def test_parse_s2_papers_non_arxiv_url():
    papers = _parse_s2_papers(SAMPLE_S2_RESPONSE["data"])
    p_no_arxiv = papers[2]
    assert "semanticscholar.org" in p_no_arxiv.url
    assert p_no_arxiv.pdf_url is None


def test_parse_s2_papers_handles_none_items():
    items = [None, SAMPLE_S2_RESPONSE["data"][0], None]
    papers = _parse_s2_papers(items)
    assert len(papers) == 1


def test_parse_s2_papers_skips_no_title():
    items = [{"paperId": "x", "title": "", "authors": [], "externalIds": {}}]
    papers = _parse_s2_papers(items)
    assert len(papers) == 0


@patch("fetcher.s2_search_fetcher._s2_get")
def test_fetch_s2_search_dedup_across_queries(mock_get):
    mock_get.return_value = SAMPLE_S2_RESPONSE
    papers = fetch_s2_search(
        queries=["query1", "query2"],
        year="2026",
        limit_per_query=10,
    )
    ids = [p.id for p in papers]
    assert len(ids) == len(set(ids))
    assert len(papers) == 3


@patch("fetcher.s2_search_fetcher._s2_get")
def test_fetch_s2_search_empty_queries(mock_get):
    papers = fetch_s2_search(queries=[], year="2026")
    assert papers == []
    mock_get.assert_not_called()


@patch("fetcher.s2_search_fetcher._s2_get")
def test_fetch_s2_search_api_failure(mock_get):
    mock_get.side_effect = Exception("network error")
    papers = fetch_s2_search(queries=["test query"], year="2026")
    assert papers == []


@patch("fetcher.s2_search_fetcher._s2_get")
def test_fetch_s2_search_passes_year(mock_get):
    mock_get.return_value = {"total": 0, "data": []}
    fetch_s2_search(queries=["test"], year="2026", limit_per_query=15)
    call_args = mock_get.call_args
    params = call_args[0][1]
    assert params["year"] == "2026"
    assert params["limit"] == "15"
    assert params["query"] == "test"
