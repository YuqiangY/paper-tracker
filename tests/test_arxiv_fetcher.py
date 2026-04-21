from unittest.mock import patch
from fetcher.arxiv_fetcher import fetch_arxiv, _parse_arxiv_response

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


@patch("fetcher.arxiv_fetcher._query_arxiv_api")
def test_fetch_arxiv_deduplicates(mock_query):
    mock_query.return_value = SAMPLE_ARXIV_XML
    categories = ["cs.CV", "cs.CV"]  # same category twice
    papers = fetch_arxiv(categories, max_results=100, lookback_days=1)
    # should deduplicate by paper id
    ids = [p.id for p in papers]
    assert len(ids) == len(set(ids))


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
