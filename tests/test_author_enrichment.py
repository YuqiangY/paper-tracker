from unittest.mock import patch
from models import Paper
from fetcher.author_enrichment import (
    enrich_authors, _merge_s2_into_enriched,
    _fetch_arxiv_html_affiliations,
)


def _paper(id="2404.12345", authors=None):
    return Paper(
        id=id, title="Test", authors=authors or ["Alice", "Bob"],
        abstract="", url="", source="arxiv", published="2026-04-17",
        categories=[],
        authors_enriched=[
            {"name": n, "affiliation": None, "h_index": None, "semantic_scholar_id": None}
            for n in (authors or ["Alice", "Bob"])
        ],
    )


def test_merge_s2_into_enriched():
    enriched = [
        {"name": "Alice", "affiliation": "MIT", "h_index": None, "semantic_scholar_id": None},
        {"name": "Bob", "affiliation": None, "h_index": None, "semantic_scholar_id": None},
    ]
    s2_authors = [
        {"name": "Alice Smith", "authorId": "111", "affiliations": ["MIT CSAIL"]},
        {"name": "Bob Jones", "authorId": "222", "affiliations": ["Stanford"]},
    ]
    h_index_map = {"111": 45, "222": 30}
    _merge_s2_into_enriched(enriched, s2_authors, h_index_map)
    assert enriched[0]["h_index"] == 45
    assert enriched[0]["semantic_scholar_id"] == "111"
    assert enriched[0]["affiliation"] == "MIT"
    assert enriched[1]["affiliation"] == "Stanford"
    assert enriched[1]["h_index"] == 30


def test_merge_preserves_arxiv_affiliation():
    enriched = [
        {"name": "Alice", "affiliation": "MIT CSAIL", "h_index": None, "semantic_scholar_id": None},
    ]
    s2_authors = [
        {"name": "Alice", "authorId": "111", "affiliations": ["Massachusetts Institute of Technology"]},
    ]
    _merge_s2_into_enriched(enriched, s2_authors, {"111": 50})
    assert enriched[0]["affiliation"] == "MIT CSAIL"
    assert enriched[0]["h_index"] == 50


@patch("fetcher.author_enrichment._enrich_affiliations_from_arxiv_html")
@patch("fetcher.author_enrichment._batch_fetch_papers")
@patch("fetcher.author_enrichment._batch_fetch_h_indices")
def test_enrich_authors_end_to_end(mock_h, mock_papers, mock_html):
    mock_html.return_value = None  # skip HTML enrichment
    mock_papers.return_value = {
        "2404.12345": {
            "authors": [
                {"name": "Alice", "authorId": "111", "affiliations": ["MIT"]},
                {"name": "Bob", "authorId": "222", "affiliations": []},
            ]
        }
    }
    mock_h.return_value = {"111": 45}
    papers = [_paper()]
    enrich_authors(papers)
    assert papers[0].authors_enriched[0]["h_index"] == 45
    assert papers[0].authors_enriched[0]["affiliation"] == "MIT"
    assert papers[0].authors_enriched[0]["semantic_scholar_id"] == "111"


@patch("fetcher.author_enrichment._enrich_affiliations_from_arxiv_html")
@patch("fetcher.author_enrichment._batch_fetch_papers")
def test_enrich_handles_s2_failure(mock_papers, mock_html):
    mock_html.return_value = None
    mock_papers.side_effect = Exception("S2 is down")
    papers = [_paper()]
    result = enrich_authors(papers)
    assert result == papers
    assert papers[0].authors_enriched[0]["h_index"] is None


def test_fetch_arxiv_html_affiliations_type1():
    html = '''<div class="ltx_authors">
    <span class="ltx_creator ltx_role_author">
    <span class="ltx_personname">Alice</span>
    <span class="ltx_author_notes">
    <span class="ltx_contact ltx_role_affiliation">
    <span class="ltx_text ltx_affiliation_institution">MIT</span>
    </span></span></span>
    <span class="ltx_author_before">, </span>
    <span class="ltx_creator ltx_role_author">
    <span class="ltx_personname">Bob</span>
    <span class="ltx_author_notes">
    <span class="ltx_contact ltx_role_affiliation">
    <span class="ltx_text ltx_affiliation_institution">Stanford University</span>
    </span></span></span>
    </div>'''
    with patch("fetcher.author_enrichment.urllib.request.urlopen") as mock_open:
        mock_resp = mock_open.return_value.__enter__.return_value
        mock_resp.read.return_value = html.encode("utf-8")
        result, raw_text = _fetch_arxiv_html_affiliations("2404.12345", timeout=10)
    assert result == ["MIT", "Stanford University"]


def test_enrich_skips_rss_papers():
    p = Paper(
        id="rss:12345", title="RSS Paper", authors=["X"], abstract="",
        url="", source="rss:HF", published="", categories=[],
    )
    result = enrich_authors([p])
    assert p.authors_enriched is None
