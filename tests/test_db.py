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


def test_get_papers_by_date():
    with tempfile.TemporaryDirectory() as tmpdir:
        db = PaperDB(os.path.join(tmpdir, "test.db"))
        p = make_paper()
        db.insert(p)
        db.update_filter_result(p.id, 8.0, "底层视觉", "摘要", "reason", ["tag"])
        papers = db.get_papers_by_date("2026-04-17", min_score=7.0)
        assert len(papers) == 1
        assert papers[0]["id"] == "2404.12345"


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
