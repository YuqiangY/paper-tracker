from unittest.mock import patch

from models import Paper
from fetcher.fulltext import parse_paper_context, attach_paper_contexts, PaperContext

HTML = """<html><body><article>
<div class="ltx_authors"><span class="ltx_creator ltx_role_author">
  <span class="ltx_personname">Alice</span>
  <span class="ltx_author_notes"><span class="ltx_contact ltx_role_affiliation">
    <span class="ltx_affiliation_institution">MIT</span></span></span></span></div>
<section class="ltx_section"><h2 class="ltx_title ltx_title_section">1 Introduction</h2><p>Intro.</p></section>
<section class="ltx_section"><h2 class="ltx_title ltx_title_section">6 Conclusion</h2>
  <p>We show strong results. Code at <a href="https://github.com/alice/sr-net">github</a>.</p></section>
</article>
<footer><a href="https://github.com/arXiv/html_feedback/issues">feedback</a>
<a href="https://github.com/brucemiller/LaTeXML">LaTeXML</a></footer></body></html>"""


def _paper(pid="2610.00001"):
    return Paper(id=pid, title="T", authors=["Alice"], abstract="A", url="", source="arxiv",
                 published="2026-10-14", categories=["cs.CV"])


def test_parse_extracts_conclusion_and_code_ignoring_boilerplate():
    ctx = parse_paper_context(HTML)
    assert ctx.conclusion.startswith("6 Conclusion")
    assert "strong results" in ctx.conclusion
    assert ctx.code_url == "https://github.com/alice/sr-net"


def test_parse_ignores_boilerplate_only_links():
    html = HTML.replace('<a href="https://github.com/alice/sr-net">github</a>', "")
    assert parse_paper_context(html).code_url is None


def test_parse_falls_back_to_discussion_section_and_truncates():
    long_text = " ".join(["word"] * 1000)
    html = f'<section><h2 class="ltx_title_section">5 Discussion</h2><p>{long_text}</p></section>'
    ctx = parse_paper_context(html)
    assert len(ctx.conclusion.split()) == 300
    assert ctx.code_url is None


def test_parse_no_conclusion_section():
    assert parse_paper_context("<html><body><p>x</p></body></html>").conclusion == ""


@patch("fetcher.fulltext.time.sleep")
@patch("fetcher.fulltext.fetch_paper_context")
def test_attach_sets_fields_and_skips_non_arxiv(mock_fetch, _sleep):
    mock_fetch.return_value = PaperContext(affiliations=["MIT"], author_raw_text="",
                                           conclusion="We conclude.", code_url="https://github.com/a/b")
    arxiv_p, rss_p = _paper(), _paper("rss:abc")
    n = attach_paper_contexts([arxiv_p, rss_p], timeout=5)
    assert n == 1
    assert arxiv_p.html_fetched and arxiv_p.conclusion == "We conclude."
    assert arxiv_p.code_url == "https://github.com/a/b"
    assert arxiv_p.authors_enriched[0]["affiliation"] == "MIT"
    assert rss_p.html_fetched is False
    mock_fetch.assert_called_once_with("2610.00001", 5)


@patch("fetcher.fulltext.time.sleep")
@patch("fetcher.fulltext.fetch_paper_context", return_value=None)
def test_attach_no_html_marks_fetched(_f, _sleep):
    p = _paper()
    assert attach_paper_contexts([p], timeout=5) == 0
    assert p.html_fetched is True and p.conclusion is None


@patch("fetcher.fulltext.time.sleep")
@patch("fetcher.fulltext.fetch_paper_context", side_effect=TimeoutError("slow"))
def test_attach_transient_error_leaves_unfetched(_f, _sleep):
    p = _paper()
    assert attach_paper_contexts([p], timeout=5) == 0
    assert p.html_fetched is False


@patch("fetcher.fulltext.time.sleep")
@patch("fetcher.fulltext.fetch_paper_context")
def test_attach_keeps_raw_author_text_when_unparsed(mock_fetch, _sleep):
    mock_fetch.return_value = PaperContext([], "Alice Bob Tsinghua University", "", None)
    p = _paper()
    attach_paper_contexts([p], timeout=5)
    assert p.author_raw_text == "Alice Bob Tsinghua University"


def test_parse_ignores_code_links_in_bibliography():
    html = HTML.replace('<a href="https://github.com/alice/sr-net">github</a>', "") + (
        '<section class="ltx_bibliography"><ul><li class="ltx_bibitem">'
        '<a href="https://github.com/QwenLM/Qwen3-VL">Link</a></li></ul></section>')
    assert parse_paper_context(html).code_url is None
