import json
from unittest.mock import patch, MagicMock
from output.feishu_output import generate_feishu_doc, _build_markdown, _format_author, _max_h_index


SAMPLE_PAPERS = [
    {
        "id": "2404.12345",
        "title": "Super-Resolution Method",
        "authors": '["Alice", "Bob"]',
        "abstract": "About SR.",
        "url": "https://arxiv.org/abs/2404.12345",
        "relevance_score": 8.5,
        "primary_category": "底层视觉",
        "summary_zh": "一种新的超分方法",
        "tags": '["super-resolution"]',
        "authors_enriched": json.dumps([
            {"name": "Alice", "affiliation": "MIT", "h_index": 45, "semantic_scholar_id": None},
        ]),
    },
]


def test_generate_feishu_doc_calls_feishu_cli():
    with patch("output.feishu_output.subprocess.run") as mock_run:
        mock_run.return_value = MagicMock(returncode=0, stdout='{"success":true}', stderr="")
        generate_feishu_doc(papers=SAMPLE_PAPERS, date="2026-04-17")
        mock_run.assert_called_once()
        cmd = mock_run.call_args[0][0]
        assert cmd[0] == "feishu"
        assert cmd[1] == "docx"
        assert cmd[2] == "create"
        assert cmd[3] == "论文追踪 2026-04-17"
        assert "-f" in cmd


def test_generate_feishu_doc_with_wiki_space():
    with patch("output.feishu_output.subprocess.run") as mock_run:
        mock_run.return_value = MagicMock(returncode=0, stdout="{}", stderr="")
        generate_feishu_doc(papers=SAMPLE_PAPERS, date="2026-04-17", wiki_space="7542032457XXX")
        cmd = mock_run.call_args[0][0]
        assert "--wiki-space" in cmd
        assert "7542032457XXX" in cmd


def test_generate_feishu_doc_no_papers():
    with patch("output.feishu_output.subprocess.run") as mock_run:
        generate_feishu_doc(papers=[], date="2026-04-17")
        mock_run.assert_not_called()


def test_generate_feishu_doc_handles_missing_cli():
    with patch("output.feishu_output.subprocess.run", side_effect=FileNotFoundError):
        generate_feishu_doc(papers=SAMPLE_PAPERS, date="2026-04-17")


def test_build_markdown_structure():
    md = _build_markdown(SAMPLE_PAPERS, "2026-04-17")
    assert "今日概览" in md
    assert "1 篇" in md
    assert "TOP 3" in md
    assert "Super-Resolution Method" in md
    assert "底层视觉" in md
    assert "`super-resolution`" in md


def test_build_markdown_card_layout():
    md = _build_markdown(SAMPLE_PAPERS, "2026-04-17")
    assert "<callout" in md
    assert "callout>" in md
    assert "评分 **8.5**" in md
    assert "**Alice**" in md
    assert "*(MIT)*" in md
    assert "`h=45`" in md


def test_build_markdown_multiple_categories():
    papers = [
        {
            "title": "Paper A", "url": "https://a.com", "relevance_score": 9,
            "primary_category": "底层视觉", "summary_zh": "摘要A", "tags": '["sr"]',
            "authors": '["X"]',
        },
        {
            "title": "Paper B", "url": "https://b.com", "relevance_score": 7,
            "primary_category": "视频算法", "summary_zh": "摘要B", "tags": '["vqa"]',
            "authors": '["Y"]',
        },
    ]
    md = _build_markdown(papers, "2026-04-17")
    assert "底层视觉" in md
    assert "视频算法" in md
    assert "**2**" in md


def test_build_markdown_score_backgrounds():
    papers = [
        {
            "title": "High Score", "url": "https://a.com", "relevance_score": 9,
            "primary_category": "底层视觉", "summary_zh": "高分", "tags": "[]",
            "authors": '["X"]',
        },
        {
            "title": "Low Score", "url": "https://b.com", "relevance_score": 7,
            "primary_category": "底层视觉", "summary_zh": "低分", "tags": "[]",
            "authors": '["Y"]',
        },
    ]
    md = _build_markdown(papers, "2026-04-17")
    assert 'background-color="light-yellow"' in md
    assert 'background-color="light-grey"' in md


def test_build_markdown_why_relevant():
    papers = [
        {
            "title": "Paper A", "url": "https://a.com", "relevance_score": 9,
            "primary_category": "底层视觉", "summary_zh": "摘要",
            "tags": "[]", "authors": '["X"]',
            "why_relevant": "直接相关超分辨率研究",
        },
    ]
    md = _build_markdown(papers, "2026-04-17")
    assert "推荐理由" in md
    assert "直接相关超分辨率研究" in md


def test_format_author_plain():
    paper = {
        "authors_enriched": json.dumps([
            {"name": "Alice", "affiliation": "MIT", "h_index": 45, "semantic_scholar_id": None},
            {"name": "Bob", "affiliation": "Stanford", "h_index": 30, "semantic_scholar_id": None},
        ]),
    }
    result = _format_author(paper)
    assert "Alice (MIT) h=45" in result
    assert "Bob (Stanford) h=30" in result


def test_format_author_rich():
    paper = {
        "authors_enriched": json.dumps([
            {"name": "Alice", "affiliation": "MIT", "h_index": 45, "semantic_scholar_id": None},
            {"name": "Bob", "affiliation": "Stanford", "h_index": 30, "semantic_scholar_id": None},
        ]),
    }
    result = _format_author(paper, rich=True)
    assert "**Alice**" in result
    assert "*(MIT)*" in result
    assert "`h=45`" in result
    assert "**Bob**" in result


def test_format_author_without_enriched():
    paper = {"authors": '["Alice", "Bob", "Charlie"]'}
    result = _format_author(paper)
    assert result == "Alice, Bob 等"


def test_format_author_enriched_no_affiliation():
    paper = {
        "authors_enriched": json.dumps([
            {"name": "Alice", "affiliation": None, "h_index": None, "semantic_scholar_id": None},
        ]),
    }
    result = _format_author(paper)
    assert result == "Alice"


def test_format_author_shows_three():
    paper = {
        "authors_enriched": json.dumps([
            {"name": "A", "affiliation": None, "h_index": None, "semantic_scholar_id": None},
            {"name": "B", "affiliation": None, "h_index": None, "semantic_scholar_id": None},
            {"name": "C", "affiliation": None, "h_index": None, "semantic_scholar_id": None},
            {"name": "D", "affiliation": None, "h_index": None, "semantic_scholar_id": None},
        ]),
    }
    result = _format_author(paper)
    assert "A" in result
    assert "B" in result
    assert "C" in result
    assert "等共 4 人" in result


def test_max_h_index():
    paper = {
        "authors_enriched": json.dumps([
            {"name": "Alice", "affiliation": "MIT", "h_index": 45, "semantic_scholar_id": None},
            {"name": "Bob", "affiliation": "Stanford", "h_index": 30, "semantic_scholar_id": None},
        ]),
    }
    assert _max_h_index(paper) == 45


def test_max_h_index_none():
    paper = {
        "authors_enriched": json.dumps([
            {"name": "Alice", "affiliation": None, "h_index": None, "semantic_scholar_id": None},
        ]),
    }
    assert _max_h_index(paper) is None
