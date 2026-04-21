import json
from models import Paper, AuthorInfo, papers_to_json, papers_from_json


def test_paper_creation():
    p = Paper(
        id="2404.12345",
        title="Test Paper",
        authors=["Alice", "Bob"],
        abstract="A test abstract about image restoration.",
        url="https://arxiv.org/abs/2404.12345",
        source="arxiv",
        published="2026-04-17",
        categories=["cs.CV"],
        pdf_url="https://arxiv.org/pdf/2404.12345",
    )
    assert p.id == "2404.12345"
    assert p.source == "arxiv"
    assert p.pdf_url == "https://arxiv.org/pdf/2404.12345"


def test_paper_optional_fields_default():
    p = Paper(
        id="rss-001",
        title="RSS Paper",
        authors=[],
        abstract="",
        url="https://example.com/paper",
        source="rss:HuggingFace",
        published="2026-04-17",
        categories=[],
    )
    assert p.pdf_url is None
    assert p.relevance_score is None
    assert p.summary_zh is None
    assert p.primary_category is None
    assert p.tags is None
    assert p.why_relevant is None
    assert p.authors_enriched is None
    assert p.max_h_index is None
    assert p.first_author_h_index is None
    assert p.authors_display == []


def test_papers_roundtrip_json():
    papers = [
        Paper(
            id="2404.12345",
            title="Test Paper",
            authors=["Alice"],
            abstract="Abstract here.",
            url="https://arxiv.org/abs/2404.12345",
            source="arxiv",
            published="2026-04-17",
            categories=["cs.CV"],
            pdf_url=None,
            relevance_score=8.5,
            summary_zh="测试摘要",
            primary_category="底层视觉",
            tags=["super-resolution"],
            why_relevant="directly related",
        ),
    ]
    json_str = papers_to_json(papers)
    loaded = papers_from_json(json_str)
    assert len(loaded) == 1
    assert loaded[0].id == "2404.12345"
    assert loaded[0].relevance_score == 8.5
    assert loaded[0].summary_zh == "测试摘要"
    assert loaded[0].tags == ["super-resolution"]


def test_author_info_creation():
    a = AuthorInfo(name="Alice", affiliation="MIT", h_index=45, semantic_scholar_id="12345")
    assert a.name == "Alice"
    assert a.h_index == 45
    assert a.affiliation == "MIT"


def test_paper_enriched_properties():
    p = Paper(
        id="1", title="T", authors=["A", "B"], abstract="", url="",
        source="", published="", categories=[],
        authors_enriched=[
            {"name": "A", "affiliation": "MIT", "h_index": 45, "semantic_scholar_id": None},
            {"name": "B", "affiliation": "Stanford", "h_index": 60, "semantic_scholar_id": None},
        ],
    )
    assert p.max_h_index == 60
    assert p.first_author_h_index == 45
    assert "(MIT)" in p.authors_display[0]
    assert "[h=60]" in p.authors_display[1]


def test_paper_authors_display_fallback():
    p = Paper(id="1", title="T", authors=["Alice", "Bob"], abstract="",
              url="", source="", published="", categories=[])
    assert p.authors_display == ["Alice", "Bob"]


def test_papers_roundtrip_json_with_enriched():
    papers = [Paper(
        id="1", title="T", authors=["A"], abstract="", url="",
        source="", published="", categories=[],
        authors_enriched=[{"name": "A", "affiliation": "MIT", "h_index": 45, "semantic_scholar_id": "s1"}],
    )]
    json_str = papers_to_json(papers)
    loaded = papers_from_json(json_str)
    assert loaded[0].authors_enriched[0]["h_index"] == 45


def test_papers_from_json_backward_compat():
    old_json = '[{"id":"1","title":"T","authors":["A"],"abstract":"","url":"","source":"","published":"","categories":[]}]'
    papers = papers_from_json(old_json)
    assert papers[0].authors_enriched is None


def test_papers_to_json_is_valid_json():
    papers = [
        Paper(
            id="1",
            title="T",
            authors=[],
            abstract="A",
            url="http://x",
            source="arxiv",
            published="2026-01-01",
            categories=[],
        ),
    ]
    data = json.loads(papers_to_json(papers))
    assert isinstance(data, list)
    assert data[0]["id"] == "1"
