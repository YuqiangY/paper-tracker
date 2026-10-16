import json
import os
import sqlite3
import tempfile
from models import Paper
from storage.db import PaperDB


def make_paper(id="2404.12345", title="Test Paper"):
    return Paper(
        id=id,
        title=title,
        authors=["Alice"],
        abstract="Abstract about denoising.",
        url=f"https://arxiv.org/abs/{id}",
        source="arxiv",
        published="2026-04-17",
        categories=["cs.CV"],
    )


def test_insert_and_exists():
    with tempfile.TemporaryDirectory() as tmpdir:
        db = PaperDB(os.path.join(tmpdir, "test.db"))
        p = make_paper()
        assert not db.exists(p.id)
        db.insert(p)
        assert db.exists(p.id)


def test_insert_duplicate_is_ignored():
    with tempfile.TemporaryDirectory() as tmpdir:
        db = PaperDB(os.path.join(tmpdir, "test.db"))
        p = make_paper()
        db.insert(p)
        db.insert(p)  # should not raise
        assert db.exists(p.id)


def test_update_score():
    with tempfile.TemporaryDirectory() as tmpdir:
        db = PaperDB(os.path.join(tmpdir, "test.db"))
        p = make_paper()
        db.insert(p)
        db.update_filter_result(
            paper_id=p.id,
            relevance_score=8.5,
            primary_category="底层视觉",
            summary_zh="测试摘要",
            why_relevant="related to denoising",
            tags=["denoising"],
        )
        row = db.get(p.id)
        assert row["relevance_score"] == 8.5
        assert row["summary_zh"] == "测试摘要"


def test_get_unpushed_feishu():
    with tempfile.TemporaryDirectory() as tmpdir:
        db = PaperDB(os.path.join(tmpdir, "test.db"))
        p1 = make_paper("id-1", "Paper 1")
        p2 = make_paper("id-2", "Paper 2")
        db.insert(p1)
        db.insert(p2)
        db.update_filter_result("id-1", 8.0, "底层视觉", "摘要1", "reason", ["tag"])
        db.update_filter_result("id-2", 9.0, "视频算法", "摘要2", "reason", ["tag"])
        db.mark_pushed_feishu("id-1")
        unpushed = db.get_unpushed_feishu(min_score=7.0)
        assert len(unpushed) == 1
        assert unpushed[0]["id"] == "id-2"



def test_insert_with_enriched_authors():
    with tempfile.TemporaryDirectory() as tmpdir:
        db = PaperDB(os.path.join(tmpdir, "test.db"))
        p = make_paper()
        p.authors_enriched = [{"name": "Alice", "affiliation": "MIT", "h_index": 45, "semantic_scholar_id": "111"}]
        db.insert(p)
        row = db.get(p.id)
        enriched = json.loads(row["authors_enriched"])
        assert enriched[0]["h_index"] == 45


def test_update_authors_enriched():
    with tempfile.TemporaryDirectory() as tmpdir:
        db = PaperDB(os.path.join(tmpdir, "test.db"))
        p = make_paper()
        db.insert(p)
        db.update_authors_enriched(p.id, [
            {"name": "Alice", "affiliation": "MIT", "h_index": 45, "semantic_scholar_id": "111"}
        ])
        row = db.get(p.id)
        enriched = json.loads(row["authors_enriched"])
        assert enriched[0]["affiliation"] == "MIT"


def test_migration_adds_column():
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = os.path.join(tmpdir, "test.db")
        conn = sqlite3.connect(db_path)
        conn.execute("""CREATE TABLE papers (
            id TEXT PRIMARY KEY, title TEXT NOT NULL, authors TEXT,
            abstract TEXT, url TEXT, source TEXT, published TEXT,
            categories TEXT, pdf_url TEXT, first_seen TEXT,
            relevance_score REAL, primary_category TEXT,
            summary_zh TEXT, why_relevant TEXT, tags TEXT,
            pushed_feishu INTEGER DEFAULT 0, pushed_html INTEGER DEFAULT 0
        )""")
        conn.commit()
        conn.close()
        db = PaperDB(db_path)
        p = make_paper()
        p.authors_enriched = [{"name": "Alice", "affiliation": "MIT", "h_index": 45, "semantic_scholar_id": None}]
        db.insert(p)
        row = db.get(p.id)
        assert "authors_enriched" in row.keys()


from datetime import datetime, timedelta, timezone
from storage.db import (
    STATUS_PENDING, STATUS_KEYWORD_REJECTED, STATUS_SCORED, STATUS_SCORE_FAILED,
)


def _scored(p, total=85, rel=36, nov=24, qual=25, cat="底层视觉"):
    p.relevance_score = total
    p.score_relevance, p.score_novelty, p.score_quality = rel, nov, qual
    p.primary_category = cat
    return p


def test_insert_sets_pending_status():
    with tempfile.TemporaryDirectory() as tmpdir:
        db = PaperDB(os.path.join(tmpdir, "test.db"))
        db.insert(make_paper())
        assert db.get("2404.12345")["filter_status"] == STATUS_PENDING


def test_update_scores_stores_subscores_and_status():
    with tempfile.TemporaryDirectory() as tmpdir:
        db = PaperDB(os.path.join(tmpdir, "test.db"))
        p = make_paper()
        db.insert(p)
        db.update_scores(_scored(p))
        row = db.get(p.id)
        assert (row["relevance_score"], row["score_relevance"], row["score_novelty"], row["score_quality"]) == (85, 36, 24, 25)
        assert row["primary_category"] == "底层视觉"
        assert row["filter_status"] == STATUS_SCORED
        assert row["selected"] == 0


def test_get_papers_by_date_uses_selected_on():
    with tempfile.TemporaryDirectory() as tmpdir:
        db = PaperDB(os.path.join(tmpdir, "test.db"))
        sel, unsel = make_paper("id-1"), make_paper("id-2")
        for p in (sel, unsel):
            db.insert(p)
            db.update_scores(_scored(p))
        db.mark_selected(["id-1"], "2026-10-15")
        rows = db.get_papers_by_date("2026-10-15")
        assert [r["id"] for r in rows] == ["id-1"]
        assert db.get_papers_by_date("2026-10-14") == []


def test_set_filter_status_bulk():
    with tempfile.TemporaryDirectory() as tmpdir:
        db = PaperDB(os.path.join(tmpdir, "test.db"))
        for i in ("a", "b"):
            db.insert(make_paper(i))
        db.set_filter_status(["a", "b"], STATUS_KEYWORD_REJECTED)
        assert db.get("a")["filter_status"] == STATUS_KEYWORD_REJECTED
        assert db.get("b")["filter_status"] == STATUS_KEYWORD_REJECTED


def test_get_retry_candidates_returns_pending_and_failed_in_window():
    with tempfile.TemporaryDirectory() as tmpdir:
        db = PaperDB(os.path.join(tmpdir, "test.db"))
        for i in ("pend", "fail", "done", "rej"):
            db.insert(make_paper(i))
        db.set_filter_status(["fail"], STATUS_SCORE_FAILED)
        db.update_scores(_scored(make_paper("done")))
        db.set_filter_status(["rej"], STATUS_KEYWORD_REJECTED)
        now = datetime.now(timezone.utc)
        got = db.get_retry_candidates(
            since_iso=(now - timedelta(days=14)).isoformat(),
            before_iso=(now + timedelta(seconds=1)).isoformat(),
            limit=10,
        )
        assert sorted(p.id for p in got) == ["fail", "pend"]
        assert got[0].authors == ["Alice"]
        none_before = db.get_retry_candidates(
            since_iso=(now - timedelta(days=14)).isoformat(),
            before_iso=(now - timedelta(days=1)).isoformat(),
            limit=10,
        )
        assert none_before == []


def _legacy_db(path):
    """A DB as written by the old code: 1-10 scores, no new columns, user_version 0."""
    conn = sqlite3.connect(path)
    conn.executescript("""
        CREATE TABLE papers (id TEXT PRIMARY KEY, title TEXT NOT NULL, authors TEXT, abstract TEXT,
            url TEXT, source TEXT, published TEXT, categories TEXT, pdf_url TEXT, first_seen TEXT,
            relevance_score REAL, primary_category TEXT, summary_zh TEXT, why_relevant TEXT, tags TEXT,
            pushed_feishu INTEGER DEFAULT 0, pushed_html INTEGER DEFAULT 0, authors_enriched TEXT);
        INSERT INTO papers (id, title, first_seen, relevance_score) VALUES ('old-hi', 'T', '2026-10-09T10:00:00+00:00', 8.0);
        INSERT INTO papers (id, title, first_seen, relevance_score) VALUES ('old-null', 'T', '2026-10-09T10:00:00+00:00', NULL);
    """)
    conn.commit()
    conn.close()


def test_score_migration_runs_once():
    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "legacy.db")
        _legacy_db(path)
        PaperDB(path).close()
        db = PaperDB(path)
        hi, null = db.get("old-hi"), db.get("old-null")
        assert hi["relevance_score"] == 80
        assert hi["selected"] == 1 and hi["selected_on"] == "2026-10-09"
        assert hi["filter_status"] == STATUS_SCORED
        assert null["relevance_score"] is None and null["filter_status"] is None
        assert [r["id"] for r in db.get_papers_by_date("2026-10-09")] == ["old-hi"]
