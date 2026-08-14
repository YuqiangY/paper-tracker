"""Offline tests for the full-list Paper Copilot fetcher (ICML/ICLR) and the
git-LFS media-host fallback in the shared _get_json."""
import json

import pytest

from fetcher import papercopilot_fetcher as pc
from fetcher.papercopilot_full_fetcher import _parse_full, _build_full_paper, _status_accepted


# --- Status whitelist -------------------------------------------------------

def _records():
    return [
        {"title": "Accepted poster", "status": "Poster", "author": "A;B", "abstract": "x", "site": "https://openreview.net/forum?id=p1"},
        {"title": "Accepted oral", "status": "Oral", "author": "C", "abstract": "y", "site": "https://openreview.net/forum?id=o1"},
        {"title": "Conditional poster", "status": "ICLR 2026 ConditionalPoster", "author": "D", "abstract": "z", "site": "https://openreview.net/forum?id=cp1"},
        {"title": "Rejected", "status": "Reject", "author": "E", "abstract": "r"},
        {"title": "Withdrawn", "status": "Withdraw", "author": "F", "abstract": "w"},
        {"title": "Desk rejected", "status": "Desk Reject", "author": "G", "abstract": "d"},
    ]


def test_iclr_whitelist_keeps_only_accepted():
    papers = _parse_full(_records(), "iclr", 2026, ["Poster", "Oral"])
    titles = {p.title for p in papers}
    assert titles == {"Accepted poster", "Accepted oral", "Conditional poster"}
    # rejects/withdrawals dropped
    assert "Rejected" not in titles
    assert "Withdrawn" not in titles
    assert "Desk rejected" not in titles


def test_conditional_variant_matched_by_substring():
    # "Poster" must match "ICLR 2026 ConditionalPoster"
    assert _status_accepted("ICLR 2026 ConditionalPoster", ["poster"]) is True
    assert _status_accepted("ICLR 2026 ConditionalOral", ["oral"]) is True
    assert _status_accepted("Reject", ["poster", "oral"]) is False
    assert _status_accepted(None, ["poster"]) is False


def test_icml_no_whitelist_keeps_all():
    # ICML dump is accepted-only; passing accept_statuses=None keeps everything.
    recs = [
        {"title": "P", "status": "Poster", "author": "A", "abstract": "x"},
        {"title": "S", "status": "Spotlight", "author": "B", "abstract": "y"},
    ]
    papers = _parse_full(recs, "icml", 2026, None)
    assert len(papers) == 2


# --- Field mapping ----------------------------------------------------------

def test_full_paper_field_mapping():
    item = {
        "title": "Test Paper", "status": "Poster",
        "author": "Alice Smith; Bob Jones", "abstract": "An abstract.",
        "site": "https://openreview.net/forum?id=abc", "arxiv": "",
    }
    p = _build_full_paper(item, "iclr", 2026)
    assert p.id == "iclr2026:pc:test-paper"  # no arxiv -> slug
    assert p.authors == ["Alice Smith", "Bob Jones"]
    assert p.abstract == "An abstract."
    assert p.url == "https://openreview.net/forum?id=abc"  # site used as link
    assert p.source == "iclr"           # NOT "award"
    assert p.primary_category is None    # left for LLM classification, no 🏆
    assert p.venue == "ICLR 2026"
    assert p.categories == ["cs.LG"]


def test_full_paper_id_prefers_arxiv():
    item = {"title": "T", "status": "Poster", "author": "A", "arxiv": "2503.10148"}
    p = _build_full_paper(item, "icml", 2026)
    assert p.id == "icml2026:pc:2503.10148"
    # url falls back to arxiv abstract page when no oa/site
    assert p.url == "https://arxiv.org/abs/2503.10148"


def test_full_paper_skips_untitled():
    assert _build_full_paper({"status": "Poster", "author": "A"}, "icml", 2026) is None


# --- LFS media-host fallback ------------------------------------------------

LFS_POINTER = (
    "version https://git-lfs.github.com/spec/v1\n"
    "oid sha256:deadbeef\nsize 123\n"
)


def test_get_json_falls_back_to_media_on_lfs_pointer(tmp_path, monkeypatch):
    real_json = [{"title": "Real", "status": "Poster"}]
    calls = []

    def fake_request(url, **kwargs):
        calls.append(url)
        if "raw.githubusercontent.com" in url:
            return LFS_POINTER.encode()          # raw host serves LFS pointer
        if "media.githubusercontent.com" in url:
            return json.dumps(real_json).encode()  # media host serves real bytes
        raise AssertionError(f"unexpected url {url}")

    monkeypatch.setattr(pc, "request_with_retry", fake_request)
    cache = pc.DiskCache(str(tmp_path), ttl_seconds=3600)

    raw_url = pc.RAW_URL.format(conf="iclr", year=2026)
    media_url = pc.MEDIA_URL.format(conf="iclr", year=2026)
    data = pc._get_json(raw_url, cache, timeout=10, media_url=media_url)

    assert data == real_json
    assert any("raw.githubusercontent.com" in u for u in calls)
    assert any("media.githubusercontent.com" in u for u in calls)


def test_get_json_no_media_fallback_for_plain_json(tmp_path, monkeypatch):
    real_json = [{"title": "Direct", "status": "Spotlight"}]

    def fake_request(url, **kwargs):
        assert "media.githubusercontent.com" not in url  # must not hit media host
        return json.dumps(real_json).encode()

    monkeypatch.setattr(pc, "request_with_retry", fake_request)
    cache = pc.DiskCache(str(tmp_path), ttl_seconds=3600)
    url = pc.RAW_URL.format(conf="icml", year=2026)
    data = pc._get_json(url, cache, timeout=10, media_url=pc.MEDIA_URL.format(conf="icml", year=2026))
    assert data == real_json


# --- limit slicing (via fetch entry point with stubbed network) -------------

def test_fetch_applies_limit(tmp_path, monkeypatch):
    from fetcher import papercopilot_full_fetcher as full
    recs = [{"title": f"P{i}", "status": "Poster", "author": "A", "abstract": "x"} for i in range(10)]
    monkeypatch.setattr(full, "_get_json", lambda *a, **k: recs)
    papers = full.fetch_papercopilot_full("icml", 2026, cache_dir=str(tmp_path), limit=3)
    assert len(papers) == 3
