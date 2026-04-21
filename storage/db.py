from __future__ import annotations
import json
import sqlite3
from datetime import datetime, timezone
from models import Paper


class PaperDB:
    def __init__(self, db_path: str):
        self.conn = sqlite3.connect(db_path)
        self.conn.row_factory = sqlite3.Row
        self._create_tables()

    def _create_tables(self):
        self.conn.executescript("""
            CREATE TABLE IF NOT EXISTS papers (
                id TEXT PRIMARY KEY,
                title TEXT NOT NULL,
                authors TEXT,
                abstract TEXT,
                url TEXT,
                source TEXT,
                published TEXT,
                categories TEXT,
                pdf_url TEXT,
                first_seen TEXT,
                relevance_score REAL,
                primary_category TEXT,
                summary_zh TEXT,
                why_relevant TEXT,
                tags TEXT,
                pushed_feishu INTEGER DEFAULT 0,
                pushed_html INTEGER DEFAULT 0,
                authors_enriched TEXT
            );
            CREATE INDEX IF NOT EXISTS idx_published ON papers(published);
            CREATE INDEX IF NOT EXISTS idx_category ON papers(primary_category);
        """)
        self._migrate()

    def _migrate(self):
        cursor = self.conn.execute("PRAGMA table_info(papers)")
        existing = {row[1] for row in cursor.fetchall()}
        if "authors_enriched" not in existing:
            self.conn.execute("ALTER TABLE papers ADD COLUMN authors_enriched TEXT")
            self.conn.commit()

    def exists(self, paper_id: str) -> bool:
        row = self.conn.execute(
            "SELECT 1 FROM papers WHERE id = ?", (paper_id,)
        ).fetchone()
        return row is not None

    def insert(self, paper: Paper):
        if self.exists(paper.id):
            return
        now = datetime.now(timezone.utc).isoformat()
        self.conn.execute(
            """INSERT INTO papers
               (id, title, authors, abstract, url, source, published,
                categories, pdf_url, first_seen, authors_enriched)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                paper.id,
                paper.title,
                json.dumps(paper.authors, ensure_ascii=False),
                paper.abstract,
                paper.url,
                paper.source,
                paper.published,
                json.dumps(paper.categories, ensure_ascii=False),
                paper.pdf_url,
                now,
                json.dumps(paper.authors_enriched, ensure_ascii=False) if paper.authors_enriched else None,
            ),
        )
        self.conn.commit()

    def update_authors_enriched(self, paper_id: str, authors_enriched: list[dict]):
        self.conn.execute(
            "UPDATE papers SET authors_enriched = ? WHERE id = ?",
            (json.dumps(authors_enriched, ensure_ascii=False), paper_id),
        )
        self.conn.commit()

    def update_filter_result(
        self,
        paper_id: str,
        relevance_score: float,
        primary_category: str,
        summary_zh: str,
        why_relevant: str,
        tags: list[str],
    ):
        self.conn.execute(
            """UPDATE papers
               SET relevance_score = ?, primary_category = ?,
                   summary_zh = ?, why_relevant = ?, tags = ?
               WHERE id = ?""",
            (
                relevance_score,
                primary_category,
                summary_zh,
                why_relevant,
                json.dumps(tags, ensure_ascii=False),
                paper_id,
            ),
        )
        self.conn.commit()

    def get(self, paper_id: str) -> dict | None:
        row = self.conn.execute(
            "SELECT * FROM papers WHERE id = ?", (paper_id,)
        ).fetchone()
        return dict(row) if row else None

    def get_unpushed_feishu(self, min_score: float = 0.0) -> list[dict]:
        rows = self.conn.execute(
            """SELECT * FROM papers
               WHERE pushed_feishu = 0
                 AND relevance_score IS NOT NULL
                 AND relevance_score >= ?
               ORDER BY relevance_score DESC""",
            (min_score,),
        ).fetchall()
        return [dict(r) for r in rows]

    def mark_pushed_feishu(self, paper_id: str):
        self.conn.execute(
            "UPDATE papers SET pushed_feishu = 1 WHERE id = ?", (paper_id,)
        )
        self.conn.commit()

    def mark_pushed_html(self, paper_id: str):
        self.conn.execute(
            "UPDATE papers SET pushed_html = 1 WHERE id = ?", (paper_id,)
        )
        self.conn.commit()

    def get_papers_by_date(
        self, date: str, min_score: float = 0.0
    ) -> list[dict]:
        rows = self.conn.execute(
            """SELECT * FROM papers
               WHERE first_seen LIKE ?
                 AND relevance_score IS NOT NULL
                 AND relevance_score >= ?
               ORDER BY relevance_score DESC""",
            (f"{date}%", min_score),
        ).fetchall()
        return [dict(r) for r in rows]

    def close(self):
        self.conn.close()
