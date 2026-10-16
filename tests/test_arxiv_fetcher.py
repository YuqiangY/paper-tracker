from unittest.mock import patch
from fetcher.arxiv_fetcher import fetch_arxiv, _parse_arxiv_response, _parse_total_results, CategoryStats

SAMPLE_ARXIV_XML = """<?xml version="1.0" encoding="UTF-8"?>
<feed xmlns="http://www.w3.org/2005/Atom"
      xmlns:arxiv="http://arxiv.org/schemas/atom">
  <entry>
    <id>http://arxiv.org/abs/2404.12345v1</id>
    <title>A Novel Image Super-Resolution Method</title>
    <summary>We propose a new approach to single image super-resolution
using transformer-based architecture with adaptive degradation estimation.</summary>
    <author><name>Alice Smith</name></author>
    <author><name>Bob Jones</name></author>
    <published>2026-04-17T00:00:00Z</published>
    <link href="http://arxiv.org/abs/2404.12345v1" rel="alternate" type="text/html"/>
    <link href="http://arxiv.org/pdf/2404.12345v1" title="pdf" rel="related" type="application/pdf"/>
    <arxiv:primary_category term="cs.CV"/>
    <category term="cs.CV"/>
    <category term="eess.IV"/>
  </entry>
  <entry>
    <id>http://arxiv.org/abs/2404.67890v1</id>
    <title>Transformers for NLP Tasks</title>
    <summary>A survey of transformer architectures for natural language processing.</summary>
    <author><name>Charlie Brown</name></author>
    <published>2026-04-17T00:00:00Z</published>
    <link href="http://arxiv.org/abs/2404.67890v1" rel="alternate" type="text/html"/>
    <link href="http://arxiv.org/pdf/2404.67890v1" title="pdf" rel="related" type="application/pdf"/>
    <arxiv:primary_category term="cs.CL"/>
    <category term="cs.CL"/>
  </entry>
</feed>"""


def test_parse_arxiv_response():
    papers = _parse_arxiv_response(SAMPLE_ARXIV_XML)
    assert len(papers) == 2

    p = papers[0]
    assert p.id == "2404.12345"
    assert "Super-Resolution" in p.title
    assert p.authors == ["Alice Smith", "Bob Jones"]
    assert "super-resolution" in p.abstract
    assert p.url == "http://arxiv.org/abs/2404.12345v1"
    assert p.pdf_url == "http://arxiv.org/pdf/2404.12345v1"
    assert p.source == "arxiv"
    assert "cs.CV" in p.categories
    assert "eess.IV" in p.categories
    assert p.published == "2026-04-17"


def test_parse_arxiv_extracts_id_from_url():
    papers = _parse_arxiv_response(SAMPLE_ARXIV_XML)
    assert papers[0].id == "2404.12345"
    assert papers[1].id == "2404.67890"


def test_parse_arxiv_no_affiliations():
    papers = _parse_arxiv_response(SAMPLE_ARXIV_XML)
    p = papers[0]
    assert p.authors_enriched is not None
    assert len(p.authors_enriched) == 2
    assert p.authors_enriched[0]["name"] == "Alice Smith"
    assert p.authors_enriched[0]["affiliation"] is None
    assert p.authors_enriched[0]["h_index"] is None


SAMPLE_ARXIV_XML_WITH_AFFIL = """<?xml version="1.0" encoding="UTF-8"?>
<feed xmlns="http://www.w3.org/2005/Atom"
      xmlns:arxiv="http://arxiv.org/schemas/atom">
  <entry>
    <id>http://arxiv.org/abs/2404.99999v1</id>
    <title>Test With Affiliations</title>
    <summary>Test abstract.</summary>
    <author>
      <name>Alice Smith</name>
      <arxiv:affiliation>MIT CSAIL</arxiv:affiliation>
    </author>
    <author>
      <name>Bob Jones</name>
      <arxiv:affiliation>Stanford University</arxiv:affiliation>
    </author>
    <published>2026-04-17T00:00:00Z</published>
    <link href="http://arxiv.org/abs/2404.99999v1" rel="alternate" type="text/html"/>
    <category term="cs.CV"/>
  </entry>
</feed>"""


def test_parse_arxiv_extracts_affiliations():
    papers = _parse_arxiv_response(SAMPLE_ARXIV_XML_WITH_AFFIL)
    p = papers[0]
    assert p.authors_enriched[0]["name"] == "Alice Smith"
    assert p.authors_enriched[0]["affiliation"] == "MIT CSAIL"
    assert p.authors_enriched[1]["affiliation"] == "Stanford University"


def _feed(entries: list[tuple[str, str]], total: int) -> str:
    """entries: (arxiv_id, published_date)."""
    body = "".join(
        f"""<entry><id>http://arxiv.org/abs/{pid}v1</id><title>T {pid}</title>
        <summary>S</summary><author><name>A</name></author>
        <published>{day}T12:00:00Z</published>
        <link href="http://arxiv.org/abs/{pid}v1" rel="alternate" type="text/html"/>
        <category term="cs.CV"/></entry>"""
        for pid, day in entries
    )
    return (
        '<?xml version="1.0" encoding="UTF-8"?>'
        '<feed xmlns="http://www.w3.org/2005/Atom" xmlns:arxiv="http://arxiv.org/schemas/atom" '
        'xmlns:opensearch="http://a9.com/-/spec/opensearch/1.1/">'
        f"<opensearch:totalResults>{total}</opensearch:totalResults>{body}</feed>"
    )


def test_parse_total_results():
    assert _parse_total_results(_feed([], 1234)) == 1234


@patch("fetcher.arxiv_fetcher.time.sleep")
@patch("fetcher.arxiv_fetcher._query_arxiv_api")
def test_fetch_paginates_until_older_than_cutoff(mock_query, _sleep):
    pages = {
        0: _feed([("2610.00001", "2026-10-14"), ("2610.00002", "2026-10-13")], 100),
        2: _feed([("2610.00003", "2026-10-12"), ("2610.00004", "2026-10-08")], 100),
    }
    mock_query.side_effect = lambda cat, start, max_results: pages[start]
    papers, stats = fetch_arxiv(["cs.CV"], since_date="2026-10-10", page_size=2)
    assert [p.id for p in papers] == ["2610.00001", "2610.00002", "2610.00003"]
    assert stats["cs.CV"] == CategoryStats(count=3, complete=True)
    assert mock_query.call_count == 2


@patch("fetcher.arxiv_fetcher.time.sleep")
@patch("fetcher.arxiv_fetcher._query_arxiv_api")
def test_fetch_deduplicates_across_categories(mock_query, _sleep):
    mock_query.return_value = _feed([("2610.00001", "2026-10-14"), ("2610.00009", "2026-10-01")], 2)
    papers, stats = fetch_arxiv(["cs.CV", "eess.IV", "cs.CV"], since_date="2026-10-10", page_size=200)
    assert [p.id for p in papers] == ["2610.00001"]
    assert set(stats) == {"cs.CV", "eess.IV"}


@patch("fetcher.arxiv_fetcher.time.sleep")
@patch("fetcher.arxiv_fetcher._query_arxiv_api")
def test_fetch_retries_transient_empty_page(mock_query, _sleep):
    good = _feed([("2610.00001", "2026-10-14"), ("2610.00002", "2026-10-01")], 50)
    mock_query.side_effect = [_feed([], 50), good]
    papers, stats = fetch_arxiv(["cs.CV"], since_date="2026-10-10", page_size=2)
    assert len(papers) == 1
    assert stats["cs.CV"].complete is True


@patch("fetcher.arxiv_fetcher.time.sleep")
@patch("fetcher.arxiv_fetcher._query_arxiv_api")
def test_fetch_marks_incomplete_on_persistent_empty_page(mock_query, _sleep):
    mock_query.return_value = _feed([], 0)
    papers, stats = fetch_arxiv(["cs.CV"], since_date="2026-10-10")
    assert papers == []
    assert stats["cs.CV"].complete is False
    assert "empty" in stats["cs.CV"].note


@patch("fetcher.arxiv_fetcher.time.sleep")
@patch("fetcher.arxiv_fetcher._query_arxiv_api")
def test_fetch_marks_incomplete_when_max_pages_hit(mock_query, _sleep):
    mock_query.side_effect = lambda cat, start, max_results: _feed(
        [(f"2610.{start:05d}", "2026-10-14")], 10_000)
    papers, stats = fetch_arxiv(["cs.CV"], since_date="2026-10-10", page_size=1, max_pages=3)
    assert len(papers) == 3
    assert stats["cs.CV"].complete is False
    assert "max_pages" in stats["cs.CV"].note


@patch("fetcher.arxiv_fetcher.time.sleep")
@patch("fetcher.arxiv_fetcher._query_arxiv_api", side_effect=OSError("boom"))
def test_fetch_records_category_exception(_q, _sleep):
    papers, stats = fetch_arxiv(["cs.CV"], since_date="2026-10-10")
    assert papers == []
    assert stats["cs.CV"].complete is False
    assert "boom" in stats["cs.CV"].note
