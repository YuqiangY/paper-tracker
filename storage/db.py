from __future__ import annotations
import json
import sqlite3
from datetime import datetime, timezone
from models import Paper

STATUS_PENDING = "pending"
STATUS_KEYWORD_REJECTED = "keyword_rejected"
STATUS_SCORED = "scored"
STATUS_SCORE_FAILED = "score_failed"
SCHEMA_VERSION = 1


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
        new_columns = {
            "authors_enriched": "TEXT",
            "venue": "TEXT",
            "citation_count": "INTEGER",
            "tldr": "TEXT",
            "doi": "TEXT",
            "score_relevance": "INTEGER",
            "score_novelty": "INTEGER",
            "score_quality": "INTEGER",
            "selected": "INTEGER DEFAULT 0",
            "selected_on": "TEXT",
            "filter_status": "TEXT",
        }
        added = False
        for col, col_type in new_columns.items():
            if col not in existing:
                self.conn.execute(f"ALTER TABLE papers ADD COLUMN {col} {col_type}")
                added = True
        if added:
            self.conn.commit()
        self._migrate_score_scale()

    def _migrate_score_scale(self):
        """One-time: legacy 1-10 scores -> 0-100; legacy scored rows were all selected."""
        version = self.conn.execute("PRAGMA user_version").fetchone()[0]
        if version >= SCHEMA_VERSION:
            return
        self.conn.execute(
            """UPDATE papers SET relevance_score = relevance_score * 10
               WHERE relevance_score IS NOT NULL AND relevance_score <= 10"""
        )
        self.conn.execute(
            """UPDATE papers SET selected = 1, selected_on = substr(first_seen, 1, 10),
                   filter_status = ?
               WHERE relevance_score IS NOT NULL""",
            (STATUS_SCORED,),
        )
        self.conn.execute(f"PRAGMA user_version = {SCHEMA_VERSION}")
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
                categories, pdf_url, first_seen, authors_enriched,
                venue, citation_count, tldr, doi, filter_status)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
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
                paper.venue,
                paper.citation_count,
                paper.tldr,
                paper.doi,
                STATUS_PENDING,
            ),
        )
        self.conn.commit()

    def update_authors_enriched(self, paper_id: str, authors_enriched: list[dict]):
        self.conn.execute(
            "UPDATE papers SET authors_enriched = ? WHERE id = ?",
            (json.dumps(authors_enriched, ensure_ascii=False), paper_id),
        )
        self.conn.commit()

    def update_metadata(self, paper_id: str, venue: str | None,
                        citation_count: int | None, tldr: str | None,
                        doi: str | None):
        self.conn.execute(
            """UPDATE papers SET venue = ?, citation_count = ?, tldr = ?, doi = ?
               WHERE id = ?""",
            (venue, citation_count, tldr, doi, paper_id),
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

    def update_scores(self, paper: Paper):
        self.conn.execute(
            """UPDATE papers
               SET relevance_score = ?, score_relevance = ?, score_novelty = ?,
                   score_quality = ?, primary_category = ?, filter_status = ?
               WHERE id = ?""",
            (paper.relevance_score, paper.score_relevance, paper.score_novelty,
             paper.score_quality, paper.primary_category, STATUS_SCORED, paper.id),
        )
        self.conn.commit()

    def set_filter_status(self, paper_ids: list[str], status: str):
        self.conn.executemany(
            "UPDATE papers SET filter_status = ? WHERE id = ?",
            [(status, pid) for pid in paper_ids],
        )
        self.conn.commit()

    def mark_selected(self, paper_ids: list[str], selected_on: str):
        self.conn.executemany(
            "UPDATE papers SET selected = 1, selected_on = ? WHERE id = ?",
            [(selected_on, pid) for pid in paper_ids],
        )
        self.conn.commit()

    def requeue_scored(self, paper_ids: list[str]):
        """Undo this run's scoring/selection so the papers are retried next run."""
        self.conn.executemany(
            """UPDATE papers SET filter_status = ?, selected = 0, selected_on = NULL
               WHERE id = ? AND filter_status = ?""",
            [(STATUS_SCORE_FAILED, pid, STATUS_SCORED) for pid in paper_ids],
        )
        self.conn.commit()

    def get_retry_candidates(self, since_iso: str, before_iso: str, limit: int) -> list[Paper]:
        rows = self.conn.execute(
            """SELECT * FROM papers
               WHERE filter_status IN (?, ?)
                 AND first_seen >= ? AND first_seen < ?
               ORDER BY first_seen DESC
               LIMIT ?""",
            (STATUS_PENDING, STATUS_SCORE_FAILED, since_iso, before_iso, limit),
        ).fetchall()
        return [_row_to_paper(r) for r in rows]

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
               WHERE selected = 1 AND selected_on = ?
                 AND relevance_score IS NOT NULL
                 AND relevance_score >= ?
               ORDER BY relevance_score DESC""",
            (date, min_score),
        ).fetchall()
        return [dict(r) for r in rows]

    def close(self):
        self.conn.close()


def _row_to_paper(row: sqlite3.Row) -> Paper:
    def _load(value):
        return json.loads(value) if value else None
    return Paper(
        id=row["id"], title=row["title"], authors=_load(row["authors"]) or [],
        abstract=row["abstract"] or "", url=row["url"] or "", source=row["source"] or "",
        published=row["published"] or "", categories=_load(row["categories"]) or [],
        pdf_url=row["pdf_url"], authors_enriched=_load(row["authors_enriched"]),
        venue=row["venue"], citation_count=row["citation_count"], tldr=row["tldr"], doi=row["doi"],
    )
