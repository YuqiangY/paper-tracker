"""Offline tests for the ACL Anthology XML fetcher."""
import os

from fetcher.acl_fetcher import _parse, DEFAULT_VOLUMES

FIXTURES = os.path.join(os.path.dirname(__file__), "fixtures")


def _load() -> str:
    with open(os.path.join(FIXTURES, "acl_sample.xml")) as f:
        return f.read()


def test_excludes_tutorials():
    papers = _parse(_load(), 2026, set(DEFAULT_VOLUMES))
    ids = {p.id for p in papers}
    # long.1, long.2, short.1 kept; tutorials.1 dropped
    assert ids == {"acl2026:2026.acl-long.1", "acl2026:2026.acl-long.2", "acl2026:2026.acl-short.1"}
    assert all("tutorials" not in p.id for p in papers)


def test_volume_whitelist_narrowing():
    papers = _parse(_load(), 2026, {"short"})
    assert len(papers) == 1
    assert papers[0].id == "acl2026:2026.acl-short.1"


def test_field_mapping_and_inline_markup():
    papers = _parse(_load(), 2026, {"long"})
    p = next(x for x in papers if x.id == "acl2026:2026.acl-long.1")
    # <fixed-case> flattened, no stray tags
    assert p.title == "OctoTools: A Multi-Agent Framework"
    assert p.authors == ["Pan Lu", "Bowen Chen"]
    # <i> inside abstract flattened
    assert p.abstract == "We present a framework for complex reasoning with tools."
    assert p.url == "https://aclanthology.org/2026.acl-long.1/"
    assert p.pdf_url == "https://aclanthology.org/2026.acl-long.1.pdf"
    assert p.doi == "10.18653/v1/2026.acl-long.1"
    assert p.source == "acl"
    assert p.categories == ["cs.CL"]
    assert p.venue == "ACL 2026"
    assert p.primary_category is None


def test_missing_abstract_is_empty_not_crash():
    papers = _parse(_load(), 2026, {"long"})
    p = next(x for x in papers if x.id == "acl2026:2026.acl-long.2")
    assert p.abstract == ""
    assert p.title == "A Paper With No Abstract"
    assert p.doi is None  # no <doi> element


def test_limit_slicing(tmp_path, monkeypatch):
    from fetcher import acl_fetcher as mod
    monkeypatch.setattr(mod, "_get_xml", lambda *a, **k: _load())
    papers = mod.fetch_acl(2026, cache_dir=str(tmp_path), limit=2)
    assert len(papers) == 2
