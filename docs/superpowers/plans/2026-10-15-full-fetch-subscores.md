# Full arXiv Fetch + Sub-score Ranking Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the daily tracker fetch every arXiv paper in its categories, rank keyword-matched papers with a 0–100 sub-score rubric (abstract + conclusion + affiliations + code link), keep at most 50 per interest area, and make every failure visible in the Feishu bot message.

**Architecture:** arXiv is paginated back to a cutoff derived from a persisted "last successful run" (minus a 4-day overlap). Keyword-matched papers get their arXiv HTML fetched once *before* scoring (affiliations, conclusion, code link). Scoring is split: Stage A returns only sub-scores in large area-grouped batches; selection normalizes categories, applies a threshold and a per-area cap; Stage B writes Chinese summaries only for selected papers. A `RunReport` accumulates per-stage counts and problems and is rendered into every bot message.

**Tech Stack:** Python 3.12, stdlib `urllib`/`sqlite3`/`asyncio`, `anthropic` SDK (custom base URL, model `xiaomi/mimo-v2.6-pro`), BeautifulSoup, Jinja2, pytest.

**Spec:** The "Design decisions" section below (agreed in the 2026-10-15 debugging session). Executors read it before any task.

## Design decisions (spec)

1. arXiv: no `max_results=100` cap. Paginate `cat:X` sorted by `submittedDate` desc, page size 200, until an entry is older than the cutoff. `<published>` is the v1 submission time, and announcement lags submission by up to ~3.3 days, so cutoff = `last_success_utc − overlap_days(4)`; dedup absorbs the overlap. No state → `now − lookback_days(6)`. Cutoff never older than `now − max_lookback_days(14)`.
2. arXiv API sometimes returns an empty page although `totalResults` says more exist (the most likely cause of the silent 0-paper run on 2026-10-13). Retry an empty page 3 times; if still empty, the category is *incomplete*.
3. Run state lives in `data/run_state.json`. A run is "successful" (state advances) only when every arXiv category completed, arXiv returned >0 papers, and LLM scoring did not fail wholesale.
4. Every paper inserted by dedup starts as `filter_status='pending'`. Final statuses: `keyword_rejected`, `scored`, `score_failed`. `pending`/`score_failed` papers first seen in the last 14 days are re-queued (max 200) on the next run. Legacy rows keep `filter_status NULL` and are never re-queued.
5. Score scale is 0–100 = relevance (0–40) + novelty (0–30) + quality (0–30). All three sub-scores and the total are stored for every scored paper (not only selected ones).
6. Quality is judged from abstract, conclusion (≤300 words), affiliations parsed from arXiv HTML, and code link. Missing full text must not be penalized.
7. Stage A (scoring only) batch size 50, batches grouped by first keyword-matched area; papers missing from the response get one retry round, then `score_failed` + warning. Stage B (summary_zh, why_relevant, tags) runs only on selected papers, batch 20.
8. Selection: normalize `primary_category` to an interest name or `other`; `other` is never selected; require `total >= llm_relevance_threshold` and `relevance >= min_relevance`; sort by (total, relevance, novelty) desc; keep at most `max_per_area` (50) per area per run — fixed per run, not scaled by days covered.
9. Legacy DB scores (1–10) are multiplied by 10 exactly once (guarded by `PRAGMA user_version`). Legacy scored rows get `selected=1`, `selected_on=date(first_seen)`.
10. Output queries select by `selected_on = <today>` so a paper retried from an earlier day appears in today's report.
11. Model: `xiaomi/mimo-v2.6-pro` for all LLM calls. Threshold is provisional (60) and is calibrated in Task 13 from a trial run's score distribution.
12. Bot message is sent on *every* run (including 0 papers): source counts, stage funnel, output status, warnings/errors; red header when any error. Feishu doc success is read from the CLI's JSON `success` field, not only its exit code. Deploy result is reported. `main.py` exits 1 when the report has errors.
13. `llm_filter()` stays as a facade (Stage A → threshold → Stage B) so `conference_pipeline.py` keeps working unchanged.

## Global Constraints

- Python 3.12, run with `/home/yyq/miniforge3/bin/python`; tests: `python -m pytest -q -p no:cacheprovider`.
- Work only in the worktree `/home/work/ssd1/yyq/projects/agent_harness/paper-tracker-wt` (branch `feat/full-fetch-subscores`). The main checkout is used by cron (`0 18 */2 * *`) and must not be touched until Task 13's rollout step.
- Never run `deploy.sh`, never `git push`, never send to the real Feishu webhook from the worktree. Trial runs use `--no-publish`.
- arXiv API: ≥3 s between requests, `User-Agent: PaperTracker/1.0`, endpoint `https://export.arxiv.org/api/query`.
- Model id exactly `xiaomi/mimo-v2.6-pro`.
- Sub-score maxima exactly relevance 40, novelty 30, quality 30.
- Per-area cap exactly 50; overlap 4 days; fallback lookback 6 days; max lookback 14 days; retry window 14 days, retry limit 200.
- Tests must not hit the network: patch `_query_arxiv_api`, `urlopen`, `call_llm_async`, `time.sleep`.
- Match existing style: module-level `log = logging.getLogger(__name__)`, `from __future__ import annotations`, mutable `Paper` dataclass updated in place (the codebase's established idiom).
- Commit after each task with conventional-commit messages, ending with `Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>`.

## Review Focus

1. arXiv returns an empty page in the middle of pagination → category marked incomplete, run state NOT advanced, warning in bot message (Task 4 test `test_fetch_marks_incomplete_on_persistent_empty_page`, Task 11 test `test_state_not_saved_when_arxiv_incomplete`).
2. Machine off longer than 14 days → cutoff clamped to 14 days and a warning says papers may be missing (Task 3 test `test_cutoff_clamped_to_max_lookback`, Task 11 test `test_report_warns_when_cutoff_clamped`).
3. LLM returns a category with whitespace/case differences or a non-interest name → normalized or mapped to `other`; `other` never selected and does not bypass the cap (Task 8 tests).
4. LLM returns sub-scores as strings, floats, out-of-range or garbage → coerced/clamped; garbage treated as missing and retried (Task 7 test `test_coerce_scores`).
5. Opening the real DB twice must not multiply legacy scores twice; a paper first seen yesterday and selected today appears in today's output (Task 2 tests `test_score_migration_runs_once`, `test_get_papers_by_date_uses_selected_on`).

---

## File Structure

| File | Status | Responsibility |
|---|---|---|
| `models.py` | modify | `Paper` gets sub-score, context and status-helper fields |
| `storage/db.py` | modify | new columns, one-time score migration, status/selection/retry queries |
| `storage/run_state.py` | create | load/save `run_state.json`, compute arXiv cutoff |
| `fetcher/arxiv_fetcher.py` | modify | paginated fetch with empty-page retry, per-category stats |
| `fetcher/fulltext.py` | create | fetch/parse arXiv HTML once: affiliations, conclusion, code link |
| `fetcher/author_enrichment.py` | modify | expose soup-level affiliation parser + `apply_affiliations`; skip HTML refetch |
| `filter/llm_client.py` | create | shared async LLM call + JSON-array parsing |
| `filter/llm_scoring.py` | create | Stage A sub-score scoring |
| `filter/selection.py` | create | category normalization, threshold, per-area cap |
| `filter/llm_summary.py` | create | Stage B summaries for selected papers |
| `filter/llm_filter.py` | rewrite | thin facade for `conference_pipeline.py` |
| `run_report.py` | create | `RunReport` accumulator + status lines |
| `output/feishu_bot.py` | modify | status lines, alert header, empty-summary card |
| `output/feishu_output.py` | modify | return status string; 0–100 thresholds; sub-score display |
| `output/summary.py` | modify | `max_tokens` 8192 + truncation warning |
| `templates/daily.html.j2` | modify | integer score + sub-score tooltip |
| `fetcher/s2_search_fetcher.py` | modify | fix "skipping remaining N queries" count |
| `main.py` | modify | new orchestration, CLI flags, exit code |
| `config.yaml` | modify | new arXiv/filter keys, model |
| `re_enrich.py` | modify | model from config |
| `CLAUDE.md` | modify | pipeline description |

---

### Task 1: Green baseline (fix 8 stale tests)

Current baseline: 136 passed, 8 failed — all pre-existing and unrelated to this plan. Fix them so later regressions are visible.

**Files:**
- Modify: `tests/test_db.py:73-81`, `tests/test_hf_daily_fetcher.py` (sample + integration test), `tests/test_models.py:80-92`, `tests/test_llm_filter.py:80-150`, `tests/test_main.py`, `main.py:288-297`

**Interfaces:** none produced.

- [ ] **Step 1: Fix `test_get_papers_by_date` (it queried a hard-coded 2026-04-17 while `first_seen` is now).** Replace the query date:

```python
def test_get_papers_by_date():
    from datetime import datetime, timezone
    with tempfile.TemporaryDirectory() as tmpdir:
        db = PaperDB(os.path.join(tmpdir, "test.db"))
        p = make_paper()
        db.insert(p)
        db.update_filter_result(p.id, 8.0, "底层视觉", "摘要", "reason", ["tag"])
        today_utc = datetime.now(timezone.utc).date().isoformat()
        papers = db.get_papers_by_date(today_utc, min_score=7.0)
        assert len(papers) == 1
        assert papers[0]["id"] == "2404.12345"
```

(Task 2 rewrites this test again for `selected_on`.)

- [ ] **Step 2: Fix `test_fetch_hf_daily_integration` (fixture dates aged past the 30-day lookback).** Open `tests/test_hf_daily_fetcher.py`, find `SAMPLE_HF_RESPONSE`, and in the integration test rewrite every entry's `publishedAt` to today before calling:

```python
@patch("fetcher.hf_daily_fetcher._query_hf_api")
def test_fetch_hf_daily_integration(mock_api):
    import copy
    from datetime import datetime, timezone
    today = datetime.now(timezone.utc).strftime("%Y-%m-%dT00:00:00.000Z")
    sample = copy.deepcopy(SAMPLE_HF_RESPONSE)
    for entry in sample:
        entry["publishedAt"] = today
    mock_api.return_value = sample
    papers = fetch_hf_daily(lookback_days=30)
    assert len(papers) == 3
    mock_api.assert_called_once()
```

If `SAMPLE_HF_RESPONSE` contains duplicate paper ids, keep the expected count the test already asserts (3).

- [ ] **Step 3: Fix `test_paper_enriched_properties` (`authors_display` intentionally no longer shows affiliation; affiliations are exposed via `affiliations_unique`).**

```python
    assert p.max_h_index == 60
    assert p.first_author_h_index == 45
    assert p.authors_display[0] == "A [h=45]"
    assert "[h=60]" in p.authors_display[1]
    assert p.affiliations_unique == ["MIT", "Stanford"]
```

- [ ] **Step 4: Fix the three `test_llm_filter.py` failures.** `_call_claude` compares `message.usage.output_tokens >= 16384`, which fails on a `MagicMock`; and `test_llm_filter_end_to_end` patches the sync `_call_claude` but `llm_filter` uses `_call_claude_async`, so it hit the real API. In both `_call_claude` tests add after building `mock_msg`:

```python
    mock_msg.usage.input_tokens = 10
    mock_msg.usage.output_tokens = 10
```

and change the end-to-end test decorator and mock to the async function:

```python
@patch("filter.llm_filter._call_claude_async")
def test_llm_filter_end_to_end(mock_call):
    async def fake(*args, **kwargs):
        return json.dumps([...same two dicts as before...])
    mock_call.side_effect = fake
```

(Keep the two result dicts exactly as in the current test body.)

- [ ] **Step 5: Fix `test_main.py` (two failures: `run_pipeline` never creates a caller-supplied `data_dir`; also both tests make real HF/S2 network calls).** In `main.py:run_pipeline` after resolving paths add:

```python
    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(site_dir, exist_ok=True)
```

In both pipeline tests add patches so no network is used:

```python
@patch("main.fetch_s2_search", return_value=[])
@patch("main.fetch_hf_daily", return_value=[])
@patch("main.fetch_arxiv")
@patch("main.fetch_rss")
@patch("main.llm_filter")
def test_run_pipeline_saves_json(mock_llm, mock_rss, mock_arxiv, _hf, _s2):
```

and inside each test, before `run_pipeline`, disable publishing side effects:

```python
        config["output"]["html"]["deploy_cmd"] = ""
        config["output"]["feishu"]["enabled"] = False
        config["output"]["feishu_bot"]["enabled"] = False
        config["summary"]["enabled"] = False
```

(`test_run_pipeline_with_enrichment` gets the same two extra patches as its outermost decorators and the same config lines.)

- [ ] **Step 6: Run the full suite.**

Run: `cd /home/work/ssd1/yyq/projects/agent_harness/paper-tracker-wt && /home/yyq/miniforge3/bin/python -m pytest -q -p no:cacheprovider`
Expected: `144 passed`, 0 failed. If any other test is slow because it hits the network, note it in the commit body but do not fix it here.

- [ ] **Step 7: Commit.**

```bash
git add tests/ main.py
git commit -m "test: fix 8 stale tests for a green baseline"
```

---

### Task 2: Paper fields + DB schema, migration and status queries

**Files:**
- Modify: `models.py:14-34`, `storage/db.py`
- Test: `tests/test_db.py`, `tests/test_models.py`

**Interfaces:**
- Produces (`models.Paper` new fields, all defaulted): `score_relevance: int | None`, `score_novelty: int | None`, `score_quality: int | None`, `conclusion: str | None`, `code_url: str | None`, `html_fetched: bool = False`, `author_raw_text: str | None`.
- Produces (`storage/db.py`): constants `STATUS_PENDING="pending"`, `STATUS_KEYWORD_REJECTED="keyword_rejected"`, `STATUS_SCORED="scored"`, `STATUS_SCORE_FAILED="score_failed"`; methods `update_scores(paper: Paper) -> None` (sets status scored), `set_filter_status(paper_ids: list[str], status: str) -> None`, `mark_selected(paper_ids: list[str], selected_on: str) -> None`, `get_retry_candidates(since_iso: str, before_iso: str, limit: int) -> list[Paper]`; `get_papers_by_date(date, min_score=0.0)` now filters `selected = 1 AND selected_on = date`.

- [ ] **Step 1: Write failing tests** (append to `tests/test_db.py`; also delete the Task-1 version of `test_get_papers_by_date`):

```python
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
        assert got[0].authors == ["Alice"]          # JSON columns decoded
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
        db = PaperDB(path)  # second open must not multiply again
        hi, null = db.get("old-hi"), db.get("old-null")
        assert hi["relevance_score"] == 80
        assert hi["selected"] == 1 and hi["selected_on"] == "2026-10-09"
        assert hi["filter_status"] == STATUS_SCORED
        assert null["relevance_score"] is None and null["filter_status"] is None
        assert [r["id"] for r in db.get_papers_by_date("2026-10-09")] == ["old-hi"]
```

Also update `test_get_unpushed_feishu` only if it breaks (it should not: it uses `update_filter_result` and `min_score`).

- [ ] **Step 2: Run to verify failure.**

Run: `python -m pytest -q -p no:cacheprovider tests/test_db.py`
Expected: FAIL with `ImportError: cannot import name 'STATUS_PENDING'`.

- [ ] **Step 3: Add `Paper` fields** at the end of the dataclass field list in `models.py` (after `doi`):

```python
    score_relevance: int | None = None
    score_novelty: int | None = None
    score_quality: int | None = None
    conclusion: str | None = None
    code_url: str | None = None
    html_fetched: bool = False
    author_raw_text: str | None = None
```

`papers_from_json` keeps working for old JSON because every new field has a default.

- [ ] **Step 4: Implement DB changes** in `storage/db.py`.

Add constants after imports:

```python
STATUS_PENDING = "pending"
STATUS_KEYWORD_REJECTED = "keyword_rejected"
STATUS_SCORED = "scored"
STATUS_SCORE_FAILED = "score_failed"
SCHEMA_VERSION = 1
```

Extend `new_columns` in `_migrate`:

```python
            "score_relevance": "INTEGER",
            "score_novelty": "INTEGER",
            "score_quality": "INTEGER",
            "selected": "INTEGER DEFAULT 0",
            "selected_on": "TEXT",
            "filter_status": "TEXT",
```

At the end of `_migrate` call `self._migrate_score_scale()` and add:

```python
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
```

In `insert`, add `filter_status` to the column list and `STATUS_PENDING` to the values (16 placeholders).

Add methods:

```python
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
```

Module-level helper (below the class):

```python
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
```

Change `get_papers_by_date`:

```python
    def get_papers_by_date(self, date: str, min_score: float = 0.0) -> list[dict]:
        rows = self.conn.execute(
            """SELECT * FROM papers
               WHERE selected = 1 AND selected_on = ?
                 AND relevance_score IS NOT NULL
                 AND relevance_score >= ?
               ORDER BY relevance_score DESC""",
            (date, min_score),
        ).fetchall()
        return [dict(r) for r in rows]
```

- [ ] **Step 5: Run tests.**

Run: `python -m pytest -q -p no:cacheprovider tests/test_db.py tests/test_models.py`
Expected: all PASS.

- [ ] **Step 6: Commit.** `git commit -am "feat(db): sub-score columns, filter status, one-time 0-100 migration"` (add the new tests with `git add tests/test_db.py`).

---

### Task 3: Run state and arXiv cutoff

**Files:**
- Create: `storage/run_state.py`
- Test: `tests/test_run_state.py`

**Interfaces:**
- Produces: `RunState(last_success_utc: datetime | None)` (frozen dataclass); `load_run_state(path: str) -> RunState`; `save_run_state(path: str, last_success_utc: datetime, details: dict) -> None`; `compute_arxiv_cutoff(state: RunState, now: datetime, overlap_days: int, fallback_days: int, max_lookback_days: int) -> tuple[datetime, bool]` — returns `(cutoff, was_clamped)`.

- [ ] **Step 1: Write failing tests** `tests/test_run_state.py`:

```python
import json
import os
import tempfile
from datetime import datetime, timedelta, timezone

from storage.run_state import RunState, load_run_state, save_run_state, compute_arxiv_cutoff

NOW = datetime(2026, 10, 15, 10, 0, tzinfo=timezone.utc)


def test_load_missing_file_returns_empty_state():
    assert load_run_state("/nonexistent/run_state.json") == RunState(last_success_utc=None)


def test_load_corrupt_file_returns_empty_state():
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "s.json")
        with open(path, "w") as f:
            f.write("{not json")
        assert load_run_state(path).last_success_utc is None


def test_save_then_load_roundtrip():
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "s.json")
        save_run_state(path, NOW, {"arxiv": 812})
        assert load_run_state(path).last_success_utc == NOW
        with open(path) as f:
            assert json.load(f)["details"] == {"arxiv": 812}


def test_cutoff_uses_last_success_minus_overlap():
    state = RunState(last_success_utc=NOW - timedelta(days=2))
    cutoff, clamped = compute_arxiv_cutoff(state, NOW, overlap_days=4, fallback_days=6, max_lookback_days=14)
    assert cutoff == NOW - timedelta(days=6)
    assert clamped is False


def test_cutoff_falls_back_without_state():
    cutoff, clamped = compute_arxiv_cutoff(RunState(None), NOW, 4, 6, 14)
    assert cutoff == NOW - timedelta(days=6)
    assert clamped is False


def test_cutoff_clamped_to_max_lookback():
    state = RunState(last_success_utc=NOW - timedelta(days=30))
    cutoff, clamped = compute_arxiv_cutoff(state, NOW, 4, 6, 14)
    assert cutoff == NOW - timedelta(days=14)
    assert clamped is True
```

- [ ] **Step 2: Run to verify failure.** `python -m pytest -q -p no:cacheprovider tests/test_run_state.py` → `ModuleNotFoundError: storage.run_state`.

- [ ] **Step 3: Implement** `storage/run_state.py`:

```python
"""Persisted pipeline state: when the last fully successful run happened."""
from __future__ import annotations
import json
import logging
import os
from dataclasses import dataclass
from datetime import datetime, timedelta

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class RunState:
    last_success_utc: datetime | None


def load_run_state(path: str) -> RunState:
    if not os.path.exists(path):
        return RunState(last_success_utc=None)
    try:
        with open(path) as f:
            raw = json.load(f)
        return RunState(last_success_utc=datetime.fromisoformat(raw["last_success_utc"]))
    except (OSError, ValueError, KeyError, TypeError) as e:
        log.warning("Run state at %s unreadable (%s); treating as first run", path, e)
        return RunState(last_success_utc=None)


def save_run_state(path: str, last_success_utc: datetime, details: dict) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    tmp_path = f"{path}.tmp"
    with open(tmp_path, "w") as f:
        json.dump(
            {"last_success_utc": last_success_utc.isoformat(), "details": details},
            f, ensure_ascii=False, indent=2,
        )
    os.replace(tmp_path, path)


def compute_arxiv_cutoff(
    state: RunState,
    now: datetime,
    overlap_days: int,
    fallback_days: int,
    max_lookback_days: int,
) -> tuple[datetime, bool]:
    """Oldest submission time to fetch. Overlap covers the submit->announce lag."""
    if state.last_success_utc is None:
        cutoff = now - timedelta(days=fallback_days)
    else:
        cutoff = state.last_success_utc - timedelta(days=overlap_days)
    floor = now - timedelta(days=max_lookback_days)
    if cutoff < floor:
        return floor, True
    return cutoff, False
```

- [ ] **Step 4: Run tests** → all PASS.
- [ ] **Step 5: Commit.** `git add storage/run_state.py tests/test_run_state.py && git commit -m "feat: persist last successful run and derive arXiv cutoff"`

---

### Task 4: Paginated arXiv fetch with empty-page retry

**Files:**
- Modify: `fetcher/arxiv_fetcher.py:1-59`
- Test: `tests/test_arxiv_fetcher.py`

**Interfaces:**
- Produces: `CategoryStats(count: int, complete: bool, note: str = "")` (frozen dataclass); `fetch_arxiv(categories: list[str], since_date: str, page_size: int = PAGE_SIZE, max_pages: int = MAX_PAGES) -> tuple[list[Paper], dict[str, CategoryStats]]`; `_query_arxiv_api(category: str, start: int, max_results: int) -> str`; `_parse_total_results(xml_text: str) -> int`. `since_date` is `YYYY-MM-DD`; papers with `published >= since_date` are kept.

- [ ] **Step 1: Write failing tests.** Replace `test_fetch_arxiv_deduplicates` and add a feed builder:

```python
from unittest.mock import patch
from fetcher.arxiv_fetcher import fetch_arxiv, _parse_arxiv_response, _parse_total_results, CategoryStats


def _feed(entries: list[tuple[str, str]], total: int) -> str:
    """entries: (arxiv_id, published_date)."""
    body = "".join(
        f"""<entry><id>http://arxiv.org/abs/{pid}v1</id><title>T {pid}</title>
        <summary>S</summary><author><name>A</name></author>
        <published>{day}T12:00:00Z</published>
        <link href="http://arxiv.org/abs/{pid}v1" rel="alternate" type="text/html"/>
        <category term="cs.CV"/></entry>"""
        for pid, day in entries
    )
    return (
        '<?xml version="1.0" encoding="UTF-8"?>'
        '<feed xmlns="http://www.w3.org/2005/Atom" xmlns:arxiv="http://arxiv.org/schemas/atom" '
        'xmlns:opensearch="http://a9.com/-/spec/opensearch/1.1/">'
        f"<opensearch:totalResults>{total}</opensearch:totalResults>{body}</feed>"
    )


def test_parse_total_results():
    assert _parse_total_results(_feed([], 1234)) == 1234


@patch("fetcher.arxiv_fetcher.time.sleep")
@patch("fetcher.arxiv_fetcher._query_arxiv_api")
def test_fetch_paginates_until_older_than_cutoff(mock_query, _sleep):
    pages = {
        0: _feed([("2610.00001", "2026-10-14"), ("2610.00002", "2026-10-13")], 100),
        2: _feed([("2610.00003", "2026-10-12"), ("2610.00004", "2026-10-08")], 100),
    }
    mock_query.side_effect = lambda cat, start, max_results: pages[start]
    papers, stats = fetch_arxiv(["cs.CV"], since_date="2026-10-10", page_size=2)
    assert [p.id for p in papers] == ["2610.00001", "2610.00002", "2610.00003"]
    assert stats["cs.CV"] == CategoryStats(count=3, complete=True)
    assert mock_query.call_count == 2


@patch("fetcher.arxiv_fetcher.time.sleep")
@patch("fetcher.arxiv_fetcher._query_arxiv_api")
def test_fetch_deduplicates_across_categories(mock_query, _sleep):
    mock_query.return_value = _feed([("2610.00001", "2026-10-14"), ("2610.00009", "2026-10-01")], 2)
    papers, stats = fetch_arxiv(["cs.CV", "eess.IV", "cs.CV"], since_date="2026-10-10", page_size=200)
    assert [p.id for p in papers] == ["2610.00001"]
    assert set(stats) == {"cs.CV", "eess.IV"}


@patch("fetcher.arxiv_fetcher.time.sleep")
@patch("fetcher.arxiv_fetcher._query_arxiv_api")
def test_fetch_retries_transient_empty_page(mock_query, _sleep):
    good = _feed([("2610.00001", "2026-10-14"), ("2610.00002", "2026-10-01")], 50)
    mock_query.side_effect = [_feed([], 50), good]
    papers, stats = fetch_arxiv(["cs.CV"], since_date="2026-10-10", page_size=2)
    assert len(papers) == 1
    assert stats["cs.CV"].complete is True


@patch("fetcher.arxiv_fetcher.time.sleep")
@patch("fetcher.arxiv_fetcher._query_arxiv_api")
def test_fetch_marks_incomplete_on_persistent_empty_page(mock_query, _sleep):
    mock_query.return_value = _feed([], 0)   # glitch: empty first page, even total=0
    papers, stats = fetch_arxiv(["cs.CV"], since_date="2026-10-10")
    assert papers == []
    assert stats["cs.CV"].complete is False
    assert "empty" in stats["cs.CV"].note


@patch("fetcher.arxiv_fetcher.time.sleep")
@patch("fetcher.arxiv_fetcher._query_arxiv_api")
def test_fetch_marks_incomplete_when_max_pages_hit(mock_query, _sleep):
    mock_query.side_effect = lambda cat, start, max_results: _feed(
        [(f"2610.{start:05d}", "2026-10-14")], 10_000)
    papers, stats = fetch_arxiv(["cs.CV"], since_date="2026-10-10", page_size=1, max_pages=3)
    assert len(papers) == 3
    assert stats["cs.CV"].complete is False
    assert "max_pages" in stats["cs.CV"].note


@patch("fetcher.arxiv_fetcher.time.sleep")
@patch("fetcher.arxiv_fetcher._query_arxiv_api", side_effect=OSError("boom"))
def test_fetch_records_category_exception(_q, _sleep):
    papers, stats = fetch_arxiv(["cs.CV"], since_date="2026-10-10")
    assert papers == []
    assert stats["cs.CV"].complete is False
    assert "boom" in stats["cs.CV"].note
```

- [ ] **Step 2: Run to verify failure** → `ImportError: cannot import name '_parse_total_results'`.

- [ ] **Step 3: Implement.** Replace lines 1–59 of `fetcher/arxiv_fetcher.py` (keep `_parse_arxiv_response` unchanged):

```python
from __future__ import annotations
import logging
import re
import time
import urllib.parse
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from models import Paper
from .retry import request_with_retry

log = logging.getLogger(__name__)

ARXIV_API_URL = "https://export.arxiv.org/api/query"
ATOM_NS = "http://www.w3.org/2005/Atom"
ARXIV_NS = "http://arxiv.org/schemas/atom"
OPENSEARCH_NS = "http://a9.com/-/spec/opensearch/1.1/"

PAGE_SIZE = 200
MAX_PAGES = 30
EMPTY_PAGE_RETRIES = 3
REQUEST_DELAY_S = 3.0


@dataclass(frozen=True)
class CategoryStats:
    count: int
    complete: bool
    note: str = ""


def fetch_arxiv(
    categories: list[str],
    since_date: str,
    page_size: int = PAGE_SIZE,
    max_pages: int = MAX_PAGES,
) -> tuple[list[Paper], dict[str, CategoryStats]]:
    seen: dict[str, Paper] = {}
    stats: dict[str, CategoryStats] = {}
    for cat in dict.fromkeys(categories):
        try:
            papers, stats[cat] = _fetch_category(cat, since_date, page_size, max_pages)
        except Exception as e:  # noqa: BLE001 - one category must not abort the rest
            log.warning("arXiv category %s failed: %s", cat, e)
            stats[cat] = CategoryStats(count=0, complete=False, note=f"error: {e}")
            continue
        for p in papers:
            seen.setdefault(p.id, p)
        log.info("arXiv %s: %d papers since %s (complete=%s%s)", cat, stats[cat].count,
                 since_date, stats[cat].complete, f", {stats[cat].note}" if stats[cat].note else "")
    return list(seen.values()), stats


def _fetch_category(
    category: str, since_date: str, page_size: int, max_pages: int,
) -> tuple[list[Paper], CategoryStats]:
    kept: list[Paper] = []
    start = 0
    for _ in range(max_pages):
        entries, total = _fetch_page_with_retry(category, start, page_size)
        if not entries:
            if 0 < total <= start:
                return kept, CategoryStats(len(kept), True)
            return kept, CategoryStats(len(kept), False, f"empty page at start={start} (total={total})")
        kept.extend(p for p in entries if p.published >= since_date)
        start += len(entries)
        if entries[-1].published < since_date or start >= total:
            return kept, CategoryStats(len(kept), True)
    return kept, CategoryStats(len(kept), False, f"hit max_pages={max_pages}")


def _fetch_page_with_retry(category: str, start: int, page_size: int) -> tuple[list[Paper], int]:
    """arXiv intermittently serves an empty feed; retry before trusting it."""
    entries: list[Paper] = []
    total = 0
    for attempt in range(1 + EMPTY_PAGE_RETRIES):
        time.sleep(REQUEST_DELAY_S if attempt == 0 else REQUEST_DELAY_S * (attempt + 1))
        xml_text = _query_arxiv_api(category, start, page_size)
        entries, total = _parse_arxiv_response(xml_text), _parse_total_results(xml_text)
        if entries:
            return entries, total
        log.warning("arXiv %s returned empty page at start=%d (total=%d), attempt %d",
                    category, start, total, attempt + 1)
    return entries, total


def _parse_total_results(xml_text: str) -> int:
    root = ET.fromstring(xml_text)
    value = root.findtext(f"{{{OPENSEARCH_NS}}}totalResults", "0")
    try:
        return int(value.strip())
    except ValueError:
        return 0


def _query_arxiv_api(category: str, start: int, max_results: int) -> str:
    params = urllib.parse.urlencode({
        "search_query": f"cat:{category}",
        "start": start,
        "max_results": max_results,
        "sortBy": "submittedDate",
        "sortOrder": "descending",
    })
    data = request_with_retry(
        f"{ARXIV_API_URL}?{params}",
        timeout=60,
        max_attempts=5,
        base_delay=15.0,
        max_delay=180.0,
    )
    return data.decode("utf-8")
```

Note the `0 < total <= start` branch: an empty page *past* the reported total is a normal end; an empty page with `total == 0` on page one is treated as a glitch (a whole arXiv category is never empty).

- [ ] **Step 4: Run tests.** `python -m pytest -q -p no:cacheprovider tests/test_arxiv_fetcher.py` → PASS. (`main.py` still calls the old signature; it is updated in Task 11 — `tests/test_main.py` patches `main.fetch_arxiv`, so it still passes. Do not run the real pipeline until Task 11.)

- [ ] **Step 5: Commit.** `git commit -am "feat(arxiv): paginate to cutoff, retry empty pages, report per-category completeness"`

---

### Task 5: Fetch arXiv HTML once before scoring (affiliations, conclusion, code link)

**Files:**
- Create: `fetcher/fulltext.py`
- Modify: `fetcher/author_enrichment.py:101-146, 261-293`
- Test: `tests/test_fulltext.py`, `tests/test_author_enrichment.py`

**Interfaces:**
- Consumes: `Paper.html_fetched`, `Paper.author_raw_text`, `Paper.conclusion`, `Paper.code_url` (Task 2).
- Produces (`fetcher/author_enrichment.py`): `parse_affiliations_from_soup(soup: BeautifulSoup) -> tuple[list[str | None], str]`; `apply_affiliations(paper: Paper, affiliations: list[str | None]) -> None`.
- Produces (`fetcher/fulltext.py`): `PaperContext(affiliations, author_raw_text, conclusion, code_url)` frozen dataclass; `parse_paper_context(html: str) -> PaperContext`; `fetch_paper_context(arxiv_id: str, timeout: int) -> PaperContext | None` (None on HTTP 404); `attach_paper_contexts(papers: list[Paper], timeout: int, delay_s: float = 0.5) -> int` (returns number of papers that got HTML).

- [ ] **Step 1: Write failing tests** `tests/test_fulltext.py`:

```python
from unittest.mock import patch
import urllib.error

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
```

Append to `tests/test_author_enrichment.py`:

```python
@patch("fetcher.author_enrichment._fetch_arxiv_html_affiliations")
def test_html_stage_skips_prefetched_papers(mock_fetch):
    from fetcher.author_enrichment import _enrich_affiliations_from_arxiv_html
    p = Paper(id="2610.00001", title="T", authors=["A"], abstract="", url="", source="arxiv",
              published="", categories=[])
    p.html_fetched = True
    _enrich_affiliations_from_arxiv_html([p], timeout=5)
    mock_fetch.assert_not_called()
```

(Import `patch` and `Paper` at the top of the test file if not already imported.)

- [ ] **Step 2: Run to verify failure** → `ModuleNotFoundError: fetcher.fulltext`.

- [ ] **Step 3: Refactor `author_enrichment.py`.** Split `_fetch_arxiv_html_affiliations` into a soup-level parser and keep the fetch wrapper:

```python
def parse_affiliations_from_soup(soup: BeautifulSoup) -> tuple[list[str | None], str]:
    """Return (affiliations_list, raw_author_section_text); see _fetch_arxiv_html_affiliations."""
    authors_div = soup.find("div", class_="ltx_authors")
    if authors_div:
        for parse_fn in (
            _parse_structured_affiliations,
            _parse_superscript_affiliations,
            _parse_is_with_affiliations,
            _parse_team_paper_affiliations,
        ):
            result = parse_fn(authors_div)
            if result and any(a is not None for a in result):
                return result, ""
        return [], authors_div.get_text(" ", strip=True)
    return _parse_flat_text_affiliations(soup), ""


def _fetch_arxiv_html_affiliations(
    arxiv_id: str, timeout: int,
) -> tuple[list[str | None], str]:
    """Return (affiliations_list, raw_author_section_text).

    The raw text is non-empty when an ltx_authors div exists but parsing
    yielded no affiliations — callers can pass it to an LLM for extraction.
    """
    url = f"https://arxiv.org/html/{arxiv_id}"
    req = urllib.request.Request(url)
    req.add_header("User-Agent", "PaperTracker/1.0")
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        html = resp.read().decode("utf-8")
    return parse_affiliations_from_soup(BeautifulSoup(html, "html.parser"))
```

Extract the apply block from the loop into a public helper and use it in the loop:

```python
def apply_affiliations(paper: Paper, affiliations: list[str | None]) -> None:
    if not paper.authors_enriched:
        paper.authors_enriched = [
            {"name": name, "affiliation": None, "h_index": None, "semantic_scholar_id": None}
            for name in paper.authors
        ]
    unique_affils = set(a for a in affiliations if a)
    is_team_paper = len(unique_affils) == 1 and len(affiliations) < len(paper.authors_enriched)
    if is_team_paper:
        inst = next(iter(unique_affils))
        for entry in paper.authors_enriched:
            entry["affiliation"] = inst
    else:
        for idx, affil in enumerate(affiliations):
            if idx < len(paper.authors_enriched) and affil:
                paper.authors_enriched[idx]["affiliation"] = affil
```

New loop body in `_enrich_affiliations_from_arxiv_html`:

```python
    for paper in papers:
        if paper.html_fetched:
            # HTML already parsed before scoring; only the LLM fallback is left.
            if paper.author_raw_text:
                llm_pending.append((paper, paper.author_raw_text))
            continue
        try:
            affiliations, raw_text = _fetch_arxiv_html_affiliations(paper.id, timeout)
            if not affiliations:
                if raw_text:
                    llm_pending.append((paper, raw_text))
                continue
            apply_affiliations(paper, affiliations)
            log.debug("Parsed affiliations for %s from arXiv HTML", paper.id)
        except Exception as e:
            log.debug("Failed to parse arXiv HTML for %s: %s", paper.id, e)
        time.sleep(0.5)
```

- [ ] **Step 4: Implement `fetcher/fulltext.py`:**

```python
"""Fetch a paper's arXiv HTML once and extract what the scorer needs."""
from __future__ import annotations
import logging
import re
import time
import urllib.error
import urllib.request
from dataclasses import dataclass

from bs4 import BeautifulSoup
from models import Paper
from .author_enrichment import apply_affiliations, parse_affiliations_from_soup

log = logging.getLogger(__name__)

ARXIV_HTML_URL = "https://arxiv.org/html/{}"
CONCLUSION_MAX_WORDS = 300
ARXIV_ID_RE = re.compile(r"^\d{4}\.\d{4,5}$")
CODE_URL_RE = re.compile(r"^https?://(?:www\.)?(github|gitlab)\.com/([\w.-]+)/([\w.-]+)")
# Repos linked from arXiv's HTML page chrome, not from the paper.
BOILERPLATE_REPO_OWNERS = {"arxiv", "brucemiller"}
SECTION_PATTERNS = (
    re.compile(r"conclu", re.I),
    re.compile(r"summary|discussion|closing remarks", re.I),
)


@dataclass(frozen=True)
class PaperContext:
    affiliations: list[str | None]
    author_raw_text: str
    conclusion: str
    code_url: str | None


def parse_paper_context(html: str) -> PaperContext:
    soup = BeautifulSoup(html, "html.parser")
    affiliations, raw_text = parse_affiliations_from_soup(soup)
    return PaperContext(
        affiliations=affiliations,
        author_raw_text=raw_text,
        conclusion=_extract_conclusion(soup),
        code_url=_extract_code_url(soup),
    )


def fetch_paper_context(arxiv_id: str, timeout: int) -> PaperContext | None:
    req = urllib.request.Request(ARXIV_HTML_URL.format(arxiv_id))
    req.add_header("User-Agent", "PaperTracker/1.0")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            html = resp.read().decode("utf-8", errors="replace")
    except urllib.error.HTTPError as e:
        if e.code == 404:
            return None
        raise
    return parse_paper_context(html)


def attach_paper_contexts(papers: list[Paper], timeout: int, delay_s: float = 0.5) -> int:
    with_html = 0
    for paper in papers:
        if not ARXIV_ID_RE.match(paper.id):
            continue
        try:
            ctx = fetch_paper_context(paper.id, timeout)
        except Exception as e:  # noqa: BLE001 - transient; enrichment may retry later
            log.debug("arXiv HTML fetch failed for %s: %s", paper.id, e)
            time.sleep(delay_s)
            continue
        paper.html_fetched = True
        if ctx is not None:
            with_html += 1
            if any(ctx.affiliations):
                apply_affiliations(paper, ctx.affiliations)
            paper.author_raw_text = ctx.author_raw_text or None
            paper.conclusion = ctx.conclusion or None
            paper.code_url = ctx.code_url
        time.sleep(delay_s)
    log.info("arXiv HTML context: %d/%d papers have full text", with_html, len(papers))
    return with_html


def _extract_conclusion(soup: BeautifulSoup) -> str:
    headings = soup.find_all(["h2", "h3"], class_=re.compile("ltx_title_section"))
    for pattern in SECTION_PATTERNS:
        for h in headings:
            if pattern.search(h.get_text()):
                section = h.find_parent("section") or h.parent
                words = section.get_text(" ", strip=True).split()
                return " ".join(words[:CONCLUSION_MAX_WORDS])
    return ""


def _extract_code_url(soup: BeautifulSoup) -> str | None:
    for a in soup.find_all("a", href=True):
        m = CODE_URL_RE.match(a["href"])
        if m and m.group(2).lower() not in BOILERPLATE_REPO_OWNERS:
            repo = re.sub(r"\.git$", "", m.group(3).rstrip("."))
            return f"https://{m.group(1)}.com/{m.group(2)}/{repo}"
    return None
```

- [ ] **Step 5: Run tests.** `python -m pytest -q -p no:cacheprovider tests/test_fulltext.py tests/test_author_enrichment.py` → PASS.

- [ ] **Step 6: Live sanity check (network, read-only, ~10 s).**

```bash
/home/yyq/miniforge3/bin/python -c "
from fetcher.fulltext import fetch_paper_context
c = fetch_paper_context('2510.00001', 30); print(c and (c.affiliations[:3], c.code_url, c.conclusion[:120]))"
```

Expected: either `None` (no HTML for that id) or a tuple with a non-boilerplate code URL or `None`. Try two more recent ids from `data/daily/2026-10-15.json` in the main checkout if the first is `None`.

- [ ] **Step 7: Commit.** `git add fetcher/fulltext.py fetcher/author_enrichment.py tests/ && git commit -m "feat: fetch arXiv HTML once before scoring (affiliations, conclusion, code link)"`

---

### Task 6: Shared LLM client

**Files:**
- Create: `filter/llm_client.py`
- Test: `tests/test_llm_client.py`

**Interfaces:**
- Produces: `DEFAULT_MAX_TOKENS = 16384`; `async def call_llm_async(prompt: str, *, model: str, api_key: str, base_url: str | None, system: str, max_tokens: int = DEFAULT_MAX_TOKENS) -> str`; `parse_json_array(text: str) -> list[dict]`.

- [ ] **Step 1: Write failing tests** `tests/test_llm_client.py`:

```python
import asyncio
import json
from unittest.mock import AsyncMock, MagicMock, patch

from filter.llm_client import call_llm_async, parse_json_array


def _msg(blocks, stop_reason="end_turn"):
    m = MagicMock()
    m.content = blocks
    m.stop_reason = stop_reason
    m.usage.input_tokens = 10
    m.usage.output_tokens = 20
    return m


def _block(kind, text=""):
    b = MagicMock()
    b.type = kind
    b.text = text
    return b


@patch("filter.llm_client.anthropic")
def test_call_returns_first_text_block_after_thinking(mock_anthropic):
    client = mock_anthropic.AsyncAnthropic.return_value
    client.messages.create = AsyncMock(return_value=_msg([_block("thinking"), _block("text", "[1]")]))
    out = asyncio.run(call_llm_async("p", model="m", api_key="k", base_url=None, system="s"))
    assert out == "[1]"


@patch("filter.llm_client.anthropic")
def test_call_warns_on_truncation(mock_anthropic, caplog):
    client = mock_anthropic.AsyncAnthropic.return_value
    client.messages.create = AsyncMock(return_value=_msg([_block("text", "[")], stop_reason="max_tokens"))
    asyncio.run(call_llm_async("p", model="m", api_key="k", base_url=None, system="s"))
    assert "truncated" in caplog.text


def test_parse_plain_and_fenced_and_think_wrapped():
    data = [{"paper_id": "1"}]
    assert parse_json_array(json.dumps(data)) == data
    assert parse_json_array(f"Here:\n```json\n{json.dumps(data)}\n```") == data
    assert parse_json_array(f"<think>hmm</think>\n{json.dumps(data)}") == data


def test_parse_garbage_returns_empty():
    assert parse_json_array("not json") == []
    assert parse_json_array('{"a": 1}') == []
```

- [ ] **Step 2: Run to verify failure** → `ModuleNotFoundError`.

- [ ] **Step 3: Implement** `filter/llm_client.py`:

```python
"""Shared Anthropic-SDK call and JSON-array parsing for the LLM stages."""
from __future__ import annotations
import json
import logging
import re

import anthropic

log = logging.getLogger(__name__)

DEFAULT_MAX_TOKENS = 16384
_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)
_FENCE_RE = re.compile(r"```(?:json)?\s*\n?(.*?)```", re.DOTALL)


async def call_llm_async(
    prompt: str,
    *,
    model: str,
    api_key: str,
    base_url: str | None,
    system: str,
    max_tokens: int = DEFAULT_MAX_TOKENS,
) -> str:
    kwargs = {"api_key": api_key}
    if base_url:
        kwargs["base_url"] = base_url
    client = anthropic.AsyncAnthropic(**kwargs)
    message = await client.messages.create(
        model=model,
        max_tokens=max_tokens,
        system=system,
        messages=[{"role": "user", "content": prompt}],
    )
    log.info("LLM tokens: input=%d, output=%d", message.usage.input_tokens, message.usage.output_tokens)
    if message.stop_reason == "max_tokens":
        log.warning("LLM response truncated (hit max_tokens=%d)", max_tokens)
    for block in message.content:
        if block.type == "text":
            return block.text
    return ""


def parse_json_array(text: str) -> list[dict]:
    text = _THINK_RE.sub("", text).strip()
    candidates = [text]
    match = _FENCE_RE.search(text)
    if match:
        candidates.append(match.group(1).strip())
    for candidate in candidates:
        try:
            data = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(data, list):
            return [d for d in data if isinstance(d, dict)]
    return []
```

- [ ] **Step 4: Run tests** → PASS.
- [ ] **Step 5: Commit.** `git add filter/llm_client.py tests/test_llm_client.py && git commit -m "feat: shared async LLM client with truncation warning"`

---

### Task 7: Stage A — sub-score scoring

**Files:**
- Create: `filter/llm_scoring.py`
- Test: `tests/test_llm_scoring.py`

**Interfaces:**
- Consumes: `call_llm_async`, `parse_json_array` (Task 6); `Paper` score/context fields (Task 2).
- Produces: `SCORE_MAX = {"relevance": 40, "novelty": 30, "quality": 30}`; `ScoringOutcome(scored: list[Paper], failed: list[Paper])` frozen dataclass; `score_papers(papers: list[Paper], interests: list[dict], *, model: str, api_key: str, base_url: str | None, batch_size: int, area_of: dict[str, str] | None = None) -> ScoringOutcome` — sets `score_relevance/novelty/quality`, `relevance_score` (= sum, int) and raw `primary_category` on each scored paper; `build_scoring_prompt(papers: list[Paper], interests: list[dict]) -> str`; `coerce_scores(entry: dict) -> tuple[int, int, int] | None`.

- [ ] **Step 1: Write failing tests** `tests/test_llm_scoring.py`:

```python
import json
from unittest.mock import patch

from models import Paper
from filter.llm_scoring import score_papers, build_scoring_prompt, coerce_scores

INTERESTS = [{"name": "底层视觉", "description": "超分、去噪"}, {"name": "AI Agent", "description": "智能体"}]


def _paper(pid, **kw):
    return Paper(id=pid, title=f"T{pid}", authors=["A"], abstract=f"abs {pid}", url="",
                 source="arxiv", published="2026-10-14", categories=["cs.CV"], **kw)


def _entry(pid, rel=36, nov=24, qual=25, cat="底层视觉"):
    return {"paper_id": pid, "relevance": rel, "novelty": nov, "quality": qual, "primary_category": cat}


def _fake_llm(responses_by_call):
    """responses_by_call: list of callables(prompt)->list[dict], consumed in order."""
    calls = []

    async def fake(prompt, **kwargs):
        calls.append(prompt)
        return json.dumps(responses_by_call[len(calls) - 1](prompt))
    return fake, calls


def test_coerce_scores():
    assert coerce_scores(_entry("x")) == (36, 24, 25)
    assert coerce_scores({"relevance": "38", "novelty": 31.6, "quality": -3}) == (38, 30, 0)
    assert coerce_scores({"relevance": "high", "novelty": 1, "quality": 1}) is None
    assert coerce_scores({"relevance": 1, "novelty": 1}) is None


def test_prompt_contains_context_and_no_penalty_rule():
    p = _paper("1", conclusion="We conclude X.", code_url="https://github.com/a/b",
               authors_enriched=[{"name": "A", "affiliation": "MIT", "h_index": None}])
    q = _paper("2")
    prompt = build_scoring_prompt([p, q], INTERESTS)
    assert "We conclude X." in prompt and "https://github.com/a/b" in prompt and "MIT" in prompt
    assert "full text unavailable" in prompt          # paper 2 has no conclusion
    assert "0-40" in prompt and "0-30" in prompt
    assert "Do not penalize" in prompt


@patch("filter.llm_scoring.call_llm_async")
def test_score_sets_fields_and_total(mock_call):
    fake, _ = _fake_llm([lambda _p: [_entry("1"), _entry("2", 10, 5, 5, "AI Agent")]])
    mock_call.side_effect = fake
    papers = [_paper("1"), _paper("2")]
    out = score_papers(papers, INTERESTS, model="m", api_key="k", base_url=None, batch_size=50)
    assert [p.id for p in out.scored] == ["1", "2"] and out.failed == []
    p1 = out.scored[0]
    assert (p1.score_relevance, p1.score_novelty, p1.score_quality, p1.relevance_score) == (36, 24, 25, 85)
    assert out.scored[1].primary_category == "AI Agent"


@patch("filter.llm_scoring.call_llm_async")
def test_missing_ids_retried_once_then_failed(mock_call):
    fake, calls = _fake_llm([
        lambda _p: [_entry("1")],            # round 1 omits 2 and 3
        lambda _p: [_entry("2")],            # retry recovers 2 only
    ])
    mock_call.side_effect = fake
    out = score_papers([_paper("1"), _paper("2"), _paper("3")], INTERESTS,
                       model="m", api_key="k", base_url=None, batch_size=50)
    assert sorted(p.id for p in out.scored) == ["1", "2"]
    assert [p.id for p in out.failed] == ["3"]
    assert len(calls) == 2
    assert "T3" in calls[1] and "T1" not in calls[1]


@patch("filter.llm_scoring.call_llm_async")
def test_batches_grouped_by_area(mock_call):
    seen_batches = []

    async def fake(prompt, **kwargs):
        ids = [line[4:] for line in prompt.splitlines() if line.startswith("ID: ")]
        seen_batches.append(ids)
        return json.dumps([_entry(i) for i in ids])
    mock_call.side_effect = fake
    papers = [_paper("a1"), _paper("b1"), _paper("a2"), _paper("b2")]
    area_of = {"a1": "A", "a2": "A", "b1": "B", "b2": "B"}
    score_papers(papers, INTERESTS, model="m", api_key="k", base_url=None, batch_size=2, area_of=area_of)
    assert sorted(seen_batches) == [["a1", "a2"], ["b1", "b2"]]


@patch("filter.llm_scoring.call_llm_async")
def test_garbage_scores_count_as_missing(mock_call):
    fake, _ = _fake_llm([
        lambda _p: [{"paper_id": "1", "relevance": "n/a", "novelty": 1, "quality": 1}],
        lambda _p: [_entry("1")],
    ])
    mock_call.side_effect = fake
    out = score_papers([_paper("1")], INTERESTS, model="m", api_key="k", base_url=None, batch_size=50)
    assert [p.id for p in out.scored] == ["1"]
```

- [ ] **Step 2: Run to verify failure** → `ModuleNotFoundError`.

- [ ] **Step 3: Implement** `filter/llm_scoring.py`:

```python
"""Stage A: score keyword-matched papers with a 0-100 sub-score rubric (no summaries)."""
from __future__ import annotations
import asyncio
import logging
from dataclasses import dataclass

from models import Paper
from .llm_client import call_llm_async, parse_json_array

log = logging.getLogger(__name__)

MAX_CONCURRENT = 5
PARSE_ATTEMPTS = 2
SCORE_MAX = {"relevance": 40, "novelty": 30, "quality": 30}
MAX_AUTHORS_IN_PROMPT = 5

SYSTEM_MESSAGE = (
    "You are an expert reviewer in computer vision, machine learning and AI. "
    "You score papers against a fixed rubric and return strict JSON. "
    "Compare the papers within a batch so that scores are consistent with each other."
)


@dataclass(frozen=True)
class ScoringOutcome:
    scored: list[Paper]
    failed: list[Paper]


def score_papers(
    papers: list[Paper],
    interests: list[dict],
    *,
    model: str,
    api_key: str,
    base_url: str | None,
    batch_size: int,
    area_of: dict[str, str] | None = None,
) -> ScoringOutcome:
    if not papers:
        return ScoringOutcome(scored=[], failed=[])
    area_of = area_of or {}
    ordered = sorted(papers, key=lambda p: area_of.get(p.id, ""))
    batches = [ordered[i : i + batch_size] for i in range(0, len(ordered), batch_size)]
    log.info("Stage A scoring %d papers in %d batches", len(papers), len(batches))

    scored_ids = _run_round(batches, interests, model, api_key, base_url)
    missing = [p for p in ordered if p.id not in scored_ids]
    if missing:
        log.warning("Stage A: %d papers missing from LLM output, retrying once", len(missing))
        retry_batches = [missing[i : i + batch_size] for i in range(0, len(missing), batch_size)]
        scored_ids |= _run_round(retry_batches, interests, model, api_key, base_url)

    scored = [p for p in papers if p.id in scored_ids]
    failed = [p for p in papers if p.id not in scored_ids]
    if failed:
        log.warning("Stage A: %d papers could not be scored: %s",
                    len(failed), ", ".join(p.id for p in failed[:10]))
    return ScoringOutcome(scored=scored, failed=failed)


def coerce_scores(entry: dict) -> tuple[int, int, int] | None:
    values = []
    for key, upper in SCORE_MAX.items():
        try:
            value = round(float(entry[key]))
        except (KeyError, TypeError, ValueError):
            return None
        values.append(min(max(value, 0), upper))
    return tuple(values)


def _run_round(batches, interests, model, api_key, base_url) -> set[str]:
    results = asyncio.run(_score_batches(batches, interests, model, api_key, base_url))
    scored_ids: set[str] = set()
    for batch, entries in results:
        by_id = {p.id: p for p in batch}
        for entry in entries:
            paper = by_id.get(str(entry.get("paper_id", "")))
            scores = coerce_scores(entry)
            if paper is None or scores is None:
                continue
            paper.score_relevance, paper.score_novelty, paper.score_quality = scores
            paper.relevance_score = sum(scores)
            paper.primary_category = str(entry.get("primary_category") or "")
            scored_ids.add(paper.id)
    return scored_ids


async def _score_batches(batches, interests, model, api_key, base_url):
    semaphore = asyncio.Semaphore(MAX_CONCURRENT)

    async def one(batch, idx):
        async with semaphore:
            prompt = build_scoring_prompt(batch, interests)
            for attempt in range(PARSE_ATTEMPTS):
                text = await call_llm_async(prompt, model=model, api_key=api_key,
                                            base_url=base_url, system=SYSTEM_MESSAGE)
                entries = parse_json_array(text)
                if entries:
                    return batch, entries
                log.warning("Stage A batch %d/%d unparseable (attempt %d/%d)",
                            idx + 1, len(batches), attempt + 1, PARSE_ATTEMPTS)
            return batch, []

    return await asyncio.gather(*(one(b, i) for i, b in enumerate(batches)))


def build_scoring_prompt(papers: list[Paper], interests: list[dict]) -> str:
    interest_desc = "\n".join(f"- **{i['name']}**: {i.get('description', '')}" for i in interests)
    names = ", ".join(f'"{i["name"]}"' for i in interests)
    entries = "\n---\n".join(_paper_block(p) for p in papers)
    return f"""Score each paper for a researcher with these interest areas:

{interest_desc}

Rubric (integers; total = relevance + novelty + quality, max 100):
- relevance (0-40): 36-40 a core problem of one interest area; 28-35 clearly relevant;
  16-27 tangential, or a relevant method applied elsewhere; 0-15 weak or unrelated.
- novelty (0-30): 26-30 new problem formulation or paradigm; 18-25 a clearly new method
  component; 10-17 incremental combination of known parts; 0-9 minor variant or engineering report.
- quality (0-30): strength of evidence — breadth and rigor of experiments and comparisons,
  released code, scale, credible concrete results stated in the conclusion, and the
  institutions' track record in this area.
  Do not penalize a paper for missing conclusion, affiliations or code link: some papers
  have no HTML full text. Judge from what is given; with only an abstract, use the abstract.

primary_category must be exactly one of: {names}, or "other" if none fits.

Return ONLY a JSON array, one object per paper, no other text:
[{{"paper_id": "<ID>", "relevance": <int>, "novelty": <int>, "quality": <int>, "primary_category": "<name>"}}]

Papers:

{entries}"""


def _paper_block(p: Paper) -> str:
    lines = [f"ID: {p.id}", f"Title: {p.title}"]
    names = [a["name"] for a in (p.authors_enriched or [])][:MAX_AUTHORS_IN_PROMPT] or p.authors[:MAX_AUTHORS_IN_PROMPT]
    if names:
        lines.append(f"Authors: {', '.join(names)}")
    affils = list(dict.fromkeys(
        a["affiliation"] for a in (p.authors_enriched or []) if a.get("affiliation")
    ))
    if affils:
        lines.append(f"Affiliations: {'; '.join(affils[:MAX_AUTHORS_IN_PROMPT])}")
    lines.append(f"Code: {p.code_url or 'not found'}")
    lines.append(f"Abstract: {p.abstract}")
    lines.append(f"Conclusion: {p.conclusion}" if p.conclusion else "Conclusion: (full text unavailable)")
    return "\n".join(lines) + "\n"
```

Note: `Paper.authors_enriched` dicts from arXiv parsing have `name`; `test_prompt_contains_context_and_no_penalty_rule` passes `authors_enriched` through the `Paper` constructor kwargs.

- [ ] **Step 4: Run tests** → PASS.
- [ ] **Step 5: Commit.** `git add filter/llm_scoring.py tests/test_llm_scoring.py && git commit -m "feat: stage A sub-score scoring with area-grouped batches and missing-id retry"`

---

### Task 8: Selection — category normalization, threshold, per-area cap

**Files:**
- Create: `filter/selection.py`
- Test: `tests/test_selection.py`

**Interfaces:**
- Produces: `OTHER_CATEGORY = "other"`; `normalize_category(raw: str | None, interest_names: list[str]) -> str`; `normalize_categories(papers: list[Paper], interest_names: list[str]) -> int` (mutates `primary_category`, returns number mapped to other); `SelectionResult(selected: list[Paper], below_threshold: int, other_count: int, dropped_by_cap: dict[str, int])` frozen dataclass; `select_papers(papers: list[Paper], *, interest_names: list[str], threshold: float, min_relevance: int, max_per_area: int) -> SelectionResult`. Expects categories already normalized.

- [ ] **Step 1: Write failing tests** `tests/test_selection.py`:

```python
from models import Paper
from filter.selection import (
    OTHER_CATEGORY, normalize_category, normalize_categories, select_papers,
)

NAMES = ["底层视觉", "AI Agent", "多模态大模型"]


def _p(pid, total, rel=30, nov=20, cat="底层视觉"):
    p = Paper(id=pid, title=pid, authors=[], abstract="", url="", source="arxiv",
              published="", categories=[])
    p.relevance_score, p.score_relevance, p.score_novelty, p.primary_category = total, rel, nov, cat
    return p


def test_normalize_category_variants():
    assert normalize_category("底层视觉", NAMES) == "底层视觉"
    assert normalize_category(" 底层视觉 ", NAMES) == "底层视觉"
    assert normalize_category("ai agent", NAMES) == "AI Agent"
    assert normalize_category("多模态", NAMES) == "多模态大模型"        # unique substring
    assert normalize_category("low-level vision", NAMES) == OTHER_CATEGORY
    assert normalize_category("", NAMES) == OTHER_CATEGORY
    assert normalize_category(None, NAMES) == OTHER_CATEGORY


def test_normalize_categories_counts_other():
    papers = [_p("a", 80, cat=" AI Agent"), _p("b", 80, cat="robotics")]
    assert normalize_categories(papers, NAMES) == 1
    assert [p.primary_category for p in papers] == ["AI Agent", OTHER_CATEGORY]


def test_select_applies_threshold_min_relevance_and_excludes_other():
    papers = [
        _p("ok", 75), _p("low", 55), _p("offtopic", 80, rel=15),
        _p("other", 90, cat=OTHER_CATEGORY),
    ]
    r = select_papers(papers, interest_names=NAMES, threshold=60, min_relevance=24, max_per_area=50)
    assert [p.id for p in r.selected] == ["ok"]
    assert r.below_threshold == 2
    assert r.other_count == 1


def test_select_caps_per_area_with_tiebreak():
    papers = [_p(f"v{i}", 80, rel=30 + (i % 3)) for i in range(5)] + [_p("g1", 70, cat="AI Agent")]
    r = select_papers(papers, interest_names=NAMES, threshold=60, min_relevance=24, max_per_area=3)
    vision = [p.id for p in r.selected if p.primary_category == "底层视觉"]
    assert len(vision) == 3
    assert vision[0] in {"v2"}                 # rel 32 wins the tie at total 80
    assert r.dropped_by_cap == {"底层视觉": 2}
    assert "g1" in [p.id for p in r.selected]


def test_select_order_is_area_then_score():
    papers = [_p("a", 70, cat="AI Agent"), _p("b", 90), _p("c", 80)]
    r = select_papers(papers, interest_names=NAMES, threshold=60, min_relevance=24, max_per_area=50)
    assert [p.id for p in r.selected] == ["b", "c", "a"]
```

- [ ] **Step 2: Run to verify failure** → `ModuleNotFoundError`.

- [ ] **Step 3: Implement** `filter/selection.py`:

```python
"""Pick what gets pushed: valid category, score floor, at most N per interest area."""
from __future__ import annotations
import logging
from collections import defaultdict
from dataclasses import dataclass, field

from models import Paper

log = logging.getLogger(__name__)

OTHER_CATEGORY = "other"


@dataclass(frozen=True)
class SelectionResult:
    selected: list[Paper]
    below_threshold: int
    other_count: int
    dropped_by_cap: dict[str, int] = field(default_factory=dict)


def normalize_category(raw: str | None, interest_names: list[str]) -> str:
    value = (raw or "").strip()
    if not value:
        return OTHER_CATEGORY
    if value in interest_names:
        return value
    folded = value.casefold()
    for name in interest_names:
        if name.casefold() == folded:
            return name
    partial = [n for n in interest_names if folded in n.casefold() or n.casefold() in folded]
    return partial[0] if len(partial) == 1 else OTHER_CATEGORY


def normalize_categories(papers: list[Paper], interest_names: list[str]) -> int:
    other = 0
    for p in papers:
        p.primary_category = normalize_category(p.primary_category, interest_names)
        other += p.primary_category == OTHER_CATEGORY
    return other


def select_papers(
    papers: list[Paper],
    *,
    interest_names: list[str],
    threshold: float,
    min_relevance: int,
    max_per_area: int,
) -> SelectionResult:
    by_area: dict[str, list[Paper]] = defaultdict(list)
    below = other = 0
    for p in papers:
        if p.primary_category == OTHER_CATEGORY:
            other += 1
        elif (p.relevance_score or 0) < threshold or (p.score_relevance or 0) < min_relevance:
            below += 1
        else:
            by_area[p.primary_category].append(p)

    selected: list[Paper] = []
    dropped: dict[str, int] = {}
    for area in interest_names:
        ranked = sorted(
            by_area.get(area, []),
            key=lambda p: (-(p.relevance_score or 0), -(p.score_relevance or 0),
                           -(p.score_novelty or 0), p.id),
        )
        selected.extend(ranked[:max_per_area])
        if len(ranked) > max_per_area:
            dropped[area] = len(ranked) - max_per_area
    log.info("Selection: %d selected, %d below threshold, %d other, capped %s",
             len(selected), below, other, dropped or "none")
    return SelectionResult(selected=selected, below_threshold=below,
                           other_count=other, dropped_by_cap=dropped)
```

- [ ] **Step 4: Run tests** → PASS.
- [ ] **Step 5: Commit.** `git add filter/selection.py tests/test_selection.py && git commit -m "feat: category normalization, score floor and per-area cap"`

---

### Task 9: Stage B summaries + `llm_filter` facade

**Files:**
- Create: `filter/llm_summary.py`
- Rewrite: `filter/llm_filter.py`
- Test: `tests/test_llm_summary.py`; rewrite `tests/test_llm_filter.py`

**Interfaces:**
- Consumes: Tasks 6–8.
- Produces: `summarize_papers(papers: list[Paper], interests: list[dict], *, model: str, api_key: str, base_url: str | None, batch_size: int) -> int` (sets `summary_zh`, `why_relevant`, `tags`; returns count still missing after one retry); `llm_filter(papers, interests, model, threshold, batch_size, api_key, base_url) -> list[Paper]` (same signature as today; `SCORE_BATCH_SIZE = 50` used for Stage A).

- [ ] **Step 1: Write failing tests** `tests/test_llm_summary.py`:

```python
import json
from unittest.mock import patch

from models import Paper
from filter.llm_summary import summarize_papers

INTERESTS = [{"name": "底层视觉", "description": "超分"}]


def _p(pid):
    p = Paper(id=pid, title=f"T{pid}", authors=[], abstract="a", url="", source="arxiv",
              published="", categories=[], conclusion="We conclude.")
    p.primary_category = "底层视觉"
    return p


@patch("filter.llm_summary.call_llm_async")
def test_summaries_applied_and_missing_retried(mock_call):
    calls = []

    async def fake(prompt, **kw):
        calls.append(prompt)
        ids = ["1"] if len(calls) == 1 else ["2"]
        return json.dumps([{"paper_id": i, "summary_zh": f"摘要{i}", "why_relevant": "r", "tags": ["sr"]} for i in ids])
    mock_call.side_effect = fake
    papers = [_p("1"), _p("2"), _p("3")]
    missing = summarize_papers(papers, INTERESTS, model="m", api_key="k", base_url=None, batch_size=20)
    assert [p.summary_zh for p in papers] == ["摘要1", "摘要2", None]
    assert papers[0].tags == ["sr"]
    assert missing == 1
    assert "We conclude." in calls[0]
```

Rewrite `tests/test_llm_filter.py` (old tests targeted removed internals; parsing is covered by `test_llm_client.py`):

```python
from unittest.mock import patch

from models import Paper
from filter.llm_filter import llm_filter
from filter.llm_scoring import ScoringOutcome

INTERESTS = [{"name": "底层视觉", "keywords": ["sr"], "description": "超分"}]


def _p(pid, total):
    p = Paper(id=pid, title=pid, authors=[], abstract="", url="", source="arxiv",
              published="", categories=[])
    p.relevance_score, p.primary_category = total, " 底层视觉"
    return p


@patch("filter.llm_filter.summarize_papers", return_value=0)
@patch("filter.llm_filter.score_papers")
def test_facade_thresholds_normalizes_and_summarizes_passed(mock_score, mock_sum):
    hi, lo = _p("hi", 80), _p("lo", 40)
    mock_score.return_value = ScoringOutcome(scored=[hi, lo], failed=[])
    out = llm_filter([hi, lo], INTERESTS, "m", threshold=60, batch_size=20, api_key="k", base_url=None)
    assert out == [hi]
    assert hi.primary_category == "底层视觉"
    assert mock_sum.call_args.args[0] == [hi]
    assert mock_score.call_args.kwargs["batch_size"] == 50


@patch("filter.llm_filter.summarize_papers", return_value=0)
@patch("filter.llm_filter.score_papers")
def test_facade_threshold_zero_keeps_all_scored(mock_score, _sum):
    a, b = _p("a", 5), _p("b", 0)
    mock_score.return_value = ScoringOutcome(scored=[a, b], failed=[])
    assert llm_filter([a, b], INTERESTS, "m", 0, 20, "k", None) == [a, b]
```

- [ ] **Step 2: Run to verify failure** → `ModuleNotFoundError: filter.llm_summary`.

- [ ] **Step 3: Implement** `filter/llm_summary.py`:

```python
"""Stage B: Chinese summary, rationale and tags — only for papers that were selected."""
from __future__ import annotations
import asyncio
import logging

from models import Paper
from .llm_client import call_llm_async, parse_json_array

log = logging.getLogger(__name__)

MAX_CONCURRENT = 5
PARSE_ATTEMPTS = 2
SYSTEM_MESSAGE = (
    "You summarize research papers for a Chinese-speaking researcher. Return strict JSON."
)


def summarize_papers(
    papers: list[Paper],
    interests: list[dict],
    *,
    model: str,
    api_key: str,
    base_url: str | None,
    batch_size: int,
) -> int:
    pending = list(papers)
    for round_idx in range(2):
        if not pending:
            break
        if round_idx:
            log.warning("Stage B: %d papers missing summaries, retrying once", len(pending))
        batches = [pending[i : i + batch_size] for i in range(0, len(pending), batch_size)]
        asyncio.run(_summarize_batches(batches, interests, model, api_key, base_url))
        pending = [p for p in pending if not p.summary_zh]
    if pending:
        log.warning("Stage B: %d papers left without summary", len(pending))
    return len(pending)


async def _summarize_batches(batches, interests, model, api_key, base_url):
    semaphore = asyncio.Semaphore(MAX_CONCURRENT)

    async def one(batch):
        async with semaphore:
            prompt = _build_prompt(batch, interests)
            for _ in range(PARSE_ATTEMPTS):
                entries = parse_json_array(await call_llm_async(
                    prompt, model=model, api_key=api_key, base_url=base_url, system=SYSTEM_MESSAGE))
                if entries:
                    _apply(batch, entries)
                    return

    await asyncio.gather(*(one(b) for b in batches))


def _apply(batch: list[Paper], entries: list[dict]) -> None:
    by_id = {p.id: p for p in batch}
    for entry in entries:
        paper = by_id.get(str(entry.get("paper_id", "")))
        summary = str(entry.get("summary_zh") or "").strip()
        if paper is None or not summary:
            continue
        paper.summary_zh = summary
        paper.why_relevant = str(entry.get("why_relevant") or "")
        tags = entry.get("tags")
        paper.tags = [str(t) for t in tags] if isinstance(tags, list) else []


def _build_prompt(papers: list[Paper], interests: list[dict]) -> str:
    interest_desc = "\n".join(f"- **{i['name']}**: {i.get('description', '')}" for i in interests)
    blocks = []
    for p in papers:
        lines = [f"ID: {p.id}", f"Title: {p.title}", f"Area: {p.primary_category}",
                 f"Abstract: {p.abstract}"]
        if p.conclusion:
            lines.append(f"Conclusion: {p.conclusion}")
        blocks.append("\n".join(lines) + "\n")
    papers_text = "\n---\n".join(blocks)
    return f"""Interest areas:

{interest_desc}

For each paper return a JSON array with one object per paper:
[{{"paper_id": "<ID>", "summary_zh": "<一句话中文摘要，说明做了什么、效果如何>",
  "why_relevant": "<brief English reason it matters for its area>", "tags": ["<keyword>", "<keyword>"]}}]

Return ONLY the JSON array.

Papers:

{papers_text}"""
```

Rewrite `filter/llm_filter.py`:

```python
"""Facade kept for conference_pipeline.py: Stage A scoring -> threshold -> Stage B summaries."""
from __future__ import annotations
import logging

from models import Paper
from .llm_scoring import score_papers
from .llm_summary import summarize_papers
from .selection import normalize_categories

log = logging.getLogger(__name__)

SCORE_BATCH_SIZE = 50


def llm_filter(
    papers: list[Paper],
    interests: list[dict],
    model: str,
    threshold: float,
    batch_size: int,
    api_key: str,
    base_url: str | None,
) -> list[Paper]:
    if not papers:
        return []
    outcome = score_papers(papers, interests, model=model, api_key=api_key,
                           base_url=base_url, batch_size=SCORE_BATCH_SIZE)
    normalize_categories(outcome.scored, [i["name"] for i in interests])
    passed = [p for p in outcome.scored if (p.relevance_score or 0) >= threshold]
    summarize_papers(passed, interests, model=model, api_key=api_key,
                     base_url=base_url, batch_size=batch_size)
    return passed
```

- [ ] **Step 4: Run tests.** `python -m pytest -q -p no:cacheprovider tests/test_llm_summary.py tests/test_llm_filter.py tests/test_llm_scoring.py` → PASS. Then the full suite (`test_main` patches `main.llm_filter`, still fine).
- [ ] **Step 5: Commit.** `git add filter/ tests/ && git commit -m "feat: stage B summaries for selected papers; llm_filter becomes a facade"`

---

### Task 10: RunReport, bot status card, Feishu doc status

**Files:**
- Create: `run_report.py`
- Modify: `output/feishu_bot.py`, `output/feishu_output.py:220-265`
- Test: `tests/test_run_report.py`, `tests/test_feishu_bot.py`, `tests/test_feishu_output.py`

**Interfaces:**
- Produces (`run_report.py`): `RunReport` mutable dataclass with fields `date: str`, `source_counts: dict[str,int]`, `source_notes: dict[str,str]`, `new_count`, `retry_count`, `keyword_count`, `html_count`, `scored_count`, `score_failed`, `selected_count`, `summary_missing`, `other_count` (ints, default 0), `arxiv_complete: bool = True`, `dropped_by_cap: dict[str,int]`, `feishu_doc: str = "skipped"`, `deploy: str = "skipped"`, `errors: list[str]`, `warnings: list[str]`; property `is_failed -> bool`; `status_lines() -> list[str]`.
- Produces (`feishu_bot.py`): `send_daily_bot_message(webhook_url, summary_text, date, paper_count, pages_url="", status_lines: list[str] | None = None, is_alert: bool = False) -> bool`.
- Produces (`feishu_output.py`): `generate_feishu_doc(...) -> str` returning `"ok"`, `"ok (with warnings)"`, `"skipped"` or `"failed: <detail>"`.

- [ ] **Step 1: Write failing tests** `tests/test_run_report.py`:

```python
from run_report import RunReport


def test_status_lines_funnel_and_outputs():
    r = RunReport(date="2026-10-15")
    r.source_counts = {"arXiv": 812, "HF": 36, "S2": 0}
    r.new_count, r.retry_count, r.keyword_count = 760, 5, 370
    r.scored_count, r.score_failed, r.selected_count = 365, 5, 180
    r.dropped_by_cap = {"AI Agent": 50}
    r.feishu_doc, r.deploy = "ok", "ok"
    lines = r.status_lines()
    assert lines[0] == "来源: arXiv 812 | HF 36 | S2 0"
    assert "新增 760（含重试 5）→ 关键词 370 → 打分 365（失败 5）→ 入选 180" in lines[1]
    assert "截断: AI Agent -50" in lines[1]
    assert lines[2] == "输出: 网页 ok | 飞书文档 ok"
    assert r.is_failed is False


def test_errors_and_warnings_rendered():
    r = RunReport(date="d")
    r.warnings.append("arXiv cs.AI 不完整")
    r.errors.append("arXiv 返回 0 篇")
    lines = r.status_lines()
    assert "⚠️ arXiv cs.AI 不完整" in lines and "❌ arXiv 返回 0 篇" in lines
    assert r.is_failed is True
```

Append to `tests/test_feishu_bot.py`:

```python
def test_card_includes_status_and_alert_header():
    from output.feishu_bot import _build_card
    card = _build_card("", "2026-10-15", 0, "", status_lines=["来源: arXiv 0"], is_alert=True)
    assert card["header"]["template"] == "red"
    assert "运行异常" in card["header"]["title"]["content"]
    contents = [e["text"]["content"] for e in card["elements"] if e.get("tag") == "div"]
    assert any("来源: arXiv 0" in c for c in contents)
    assert not any(c == "" for c in contents)       # empty summary is skipped
```

Append to `tests/test_feishu_output.py`:

```python
from unittest.mock import patch, MagicMock


def _run_result(rc, stdout="", stderr=""):
    m = MagicMock()
    m.returncode, m.stdout, m.stderr = rc, stdout, stderr
    return m


@patch("output.feishu_output.subprocess.run")
def test_feishu_doc_rc3_with_success_json_is_ok(mock_run):
    from output.feishu_output import generate_feishu_doc
    mock_run.return_value = _run_result(3, '{"success": true, "content_written": true, "warnings": ["x"]}')
    papers = [{"title": "T", "url": "u", "relevance_score": 85, "primary_category": "底层视觉",
               "summary_zh": "s", "tags": "[]", "authors": '["A"]'}]
    assert generate_feishu_doc(papers, "2026-10-15") == "ok (with warnings)"


@patch("output.feishu_output.subprocess.run")
def test_feishu_doc_session_expired_is_failed(mock_run):
    from output.feishu_output import generate_feishu_doc
    mock_run.return_value = _run_result(1, '{"error": "Session expired (30 days)"}')
    papers = [{"title": "T", "url": "u", "relevance_score": 85, "primary_category": "c",
               "summary_zh": "s", "tags": "[]", "authors": '["A"]'}]
    assert generate_feishu_doc(papers, "2026-10-15").startswith("failed: ")


def test_feishu_doc_no_papers_skipped():
    from output.feishu_output import generate_feishu_doc
    assert generate_feishu_doc([], "2026-10-15") == "skipped"
```

- [ ] **Step 2: Run to verify failure.**

- [ ] **Step 3: Implement `run_report.py`:**

```python
"""Per-run accumulator of counts and problems, rendered into the bot message."""
from __future__ import annotations
from dataclasses import dataclass, field


@dataclass
class RunReport:
    date: str
    source_counts: dict[str, int] = field(default_factory=dict)
    source_notes: dict[str, str] = field(default_factory=dict)
    new_count: int = 0
    retry_count: int = 0
    keyword_count: int = 0
    html_count: int = 0
    scored_count: int = 0
    score_failed: int = 0
    selected_count: int = 0
    summary_missing: int = 0
    other_count: int = 0
    arxiv_complete: bool = True
    dropped_by_cap: dict[str, int] = field(default_factory=dict)
    feishu_doc: str = "skipped"
    deploy: str = "skipped"
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def is_failed(self) -> bool:
        return bool(self.errors)

    def status_lines(self) -> list[str]:
        sources = " | ".join(f"{name} {n}" for name, n in self.source_counts.items())
        funnel = (
            f"流程: 新增 {self.new_count}（含重试 {self.retry_count}）→ 关键词 {self.keyword_count}"
            f" → 打分 {self.scored_count}（失败 {self.score_failed}）→ 入选 {self.selected_count}"
        )
        if self.dropped_by_cap:
            funnel += "（截断: " + ", ".join(f"{a} -{n}" for a, n in self.dropped_by_cap.items()) + "）"
        lines = [f"来源: {sources}", funnel, f"输出: 网页 {self.deploy} | 飞书文档 {self.feishu_doc}"]
        lines.extend(f"⚠️ {w}" for w in self.warnings)
        lines.extend(f"❌ {e}" for e in self.errors)
        return lines
```

- [ ] **Step 4: Update `output/feishu_bot.py`.** New signatures:

```python
def send_daily_bot_message(
    webhook_url: str,
    summary_text: str,
    date: str,
    paper_count: int,
    pages_url: str = "",
    status_lines: list[str] | None = None,
    is_alert: bool = False,
) -> bool:
    ...
    card = _build_card(summary_text, date, paper_count, pages_url, status_lines, is_alert)
```

`_build_card(summary_text, date, paper_count, pages_url, status_lines=None, is_alert=False)`: build `elements` as before but only append the summary `div` when `summary_text` is non-empty; after the paper-count div, if `status_lines`, append `{"tag": "hr"}` and `{"tag": "div", "text": {"tag": "lark_md", "content": "\n".join(status_lines)}}`; header:

```python
    if is_alert:
        title, template = f"⚠️ Paper Tracker 运行异常 — {date}", "red"
    else:
        title, template = f"📄 Paper Tracker 今日热点 — {date}", "blue"
```

- [ ] **Step 5: Update `generate_feishu_doc`** to return a status string. Replace the `try` block result handling:

```python
    if not papers:
        log.info("No papers to publish to Feishu")
        return "skipped"
    ...
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
        status = _doc_status(result.returncode, result.stdout, result.stderr)
        log.info("Feishu doc: %s", status) if status.startswith("ok") else log.warning("Feishu doc: %s", status)
        return status
    except FileNotFoundError:
        log.warning("'feishu' CLI not found. Install via: npm install -g @mi/feishu@latest")
        return "failed: feishu CLI not found"
    except subprocess.TimeoutExpired:
        log.warning("Feishu doc creation timed out after 120s")
        return "failed: timeout after 120s"
    except Exception as e:
        log.warning("Feishu doc creation failed: %s", e)
        return f"failed: {e}"
    finally:
        os.unlink(tmp_file.name)
```

(Move `import os` to the module top.) Add:

```python
def _doc_status(returncode: int, stdout: str, stderr: str) -> str:
    """The CLI exits non-zero on soft warnings too; trust its JSON 'success' field."""
    try:
        body = json.loads(stdout) if stdout.strip() else {}
    except json.JSONDecodeError:
        body = {}
    if returncode == 0 or body.get("success") is True:
        return "ok (with warnings)" if body.get("warnings") else "ok"
    detail = body.get("error") or stderr.strip() or stdout.strip()
    return f"failed: {str(detail)[:200]}"
```

(Ensure `import json` is at the top of `feishu_output.py`.)

- [ ] **Step 6: Run tests.** `python -m pytest -q -p no:cacheprovider tests/test_run_report.py tests/test_feishu_bot.py tests/test_feishu_output.py` → PASS.
- [ ] **Step 7: Commit.** `git add run_report.py output/ tests/ && git commit -m "feat: run report in bot card; Feishu doc status from CLI JSON"`

---

### Task 11: Orchestration in `main.py` + config

**Files:**
- Modify: `main.py` (stages, `run_pipeline`, CLI), `config.yaml`
- Test: rewrite `tests/test_main.py`

**Interfaces:**
- Consumes: everything above.
- Produces: `run_pipeline(config, data_dir=None, db_path=None, site_dir=None, state_path=None, publish=True) -> RunReport`; CLI flags `--db`, `--data-dir`, `--site-dir`, `--state-file`, `--no-publish`; process exit code 1 when `report.is_failed`.

- [ ] **Step 1: Update `config.yaml`.** Replace the `sources.arxiv` block and the `filter` block:

```yaml
  arxiv:
    enabled: true
    lookback_days: 6        # used only when data/run_state.json is missing
    overlap_days: 4         # submit->announce lag; dedup absorbs re-fetched papers
    max_lookback_days: 14   # never backfill further than this
```

```yaml
filter:
  keyword_threshold: 1
  llm_model: "xiaomi/mimo-v2.6-pro"
  llm_relevance_threshold: 60   # 0-100 total; provisional, calibrated in Task 13
  min_relevance: 24             # relevance sub-score floor (0-40)
  max_per_area: 50              # per interest area per run
  score_batch_size: 50          # Stage A
  llm_batch_size: 20            # Stage B summaries
  retry_window_days: 14
  retry_limit: 200
```

- [ ] **Step 2: Write failing tests.** Replace `tests/test_main.py` entirely:

```python
import json
import os
import tempfile
from datetime import datetime, timezone
from unittest.mock import patch

from fetcher.arxiv_fetcher import CategoryStats
from filter.llm_scoring import ScoringOutcome
from main import load_config, run_pipeline
from models import Paper
from storage.run_state import save_run_state


def _paper(pid="2610.00001", abstract="About super-resolution denoising."):
    return Paper(id=pid, title=f"SR {pid}", authors=["Alice"], abstract=abstract,
                 url=f"https://arxiv.org/abs/{pid}", source="arxiv", published="2026-10-14",
                 categories=["cs.CV"])


def _config(tmpdir):
    config = load_config("config.yaml")
    config["output"]["html"]["output_dir"] = os.path.join(tmpdir, "site")
    config["output"]["html"]["deploy_cmd"] = ""
    config["output"]["feishu"]["enabled"] = False
    config["output"]["feishu_bot"]["enabled"] = False
    config["summary"]["enabled"] = False
    config["enrichment"] = {"enabled": False}
    for key in ("hf_daily", "s2_search", "rss"):
        config["sources"][key]["enabled"] = False
    return config


def _score_all(papers, interests, **kw):
    for p in papers:
        p.score_relevance, p.score_novelty, p.score_quality = 36, 24, 25
        p.relevance_score, p.primary_category = 85, "底层视觉"
    return ScoringOutcome(scored=list(papers), failed=[])


def _summarize(papers, interests, **kw):
    for p in papers:
        p.summary_zh, p.why_relevant, p.tags = "摘要", "r", ["sr"]
    return 0


def _run(tmpdir, **kw):
    return run_pipeline(
        _config(tmpdir), data_dir=os.path.join(tmpdir, "daily"), db_path=os.path.join(tmpdir, "p.db"),
        site_dir=os.path.join(tmpdir, "site"), state_path=os.path.join(tmpdir, "state.json"), **kw)


def test_load_config():
    config = load_config("config.yaml")
    assert config["filter"]["llm_model"] == "xiaomi/mimo-v2.6-pro"
    assert config["filter"]["max_per_area"] == 50


@patch("main.attach_paper_contexts", return_value=0)
@patch("main.summarize_papers", side_effect=_summarize)
@patch("main.score_papers", side_effect=_score_all)
@patch("main.fetch_arxiv")
def test_happy_path_saves_json_html_and_state(mock_arxiv, _score, _sum, _ctx):
    mock_arxiv.return_value = ([_paper()], {"cs.CV": CategoryStats(1, True)})
    with tempfile.TemporaryDirectory() as tmpdir:
        report = _run(tmpdir)
        assert report.selected_count == 1 and not report.is_failed
        today = datetime.now().date().isoformat()
        with open(os.path.join(tmpdir, "daily", f"{today}.json")) as f:
            assert json.load(f)[0]["summary_zh"] == "摘要"
        assert any(f.endswith(".html") for f in os.listdir(os.path.join(tmpdir, "site")))
        assert os.path.exists(os.path.join(tmpdir, "state.json"))


@patch("main.fetch_arxiv")
def test_arxiv_zero_is_error_and_state_not_saved(mock_arxiv):
    mock_arxiv.return_value = ([], {"cs.CV": CategoryStats(0, True)})
    with tempfile.TemporaryDirectory() as tmpdir:
        report = _run(tmpdir)
        assert report.is_failed
        assert any("arXiv" in e for e in report.errors)
        assert not os.path.exists(os.path.join(tmpdir, "state.json"))


@patch("main.attach_paper_contexts", return_value=0)
@patch("main.summarize_papers", side_effect=_summarize)
@patch("main.score_papers", side_effect=_score_all)
@patch("main.fetch_arxiv")
def test_state_not_saved_when_arxiv_incomplete(mock_arxiv, *_):
    mock_arxiv.return_value = ([_paper()], {"cs.CV": CategoryStats(1, False, "empty page at start=200")})
    with tempfile.TemporaryDirectory() as tmpdir:
        report = _run(tmpdir)
        assert any("cs.CV" in w for w in report.warnings)
        assert not os.path.exists(os.path.join(tmpdir, "state.json"))


@patch("main.attach_paper_contexts", return_value=0)
@patch("main.summarize_papers", side_effect=_summarize)
@patch("main.score_papers")
@patch("main.fetch_arxiv")
def test_failed_scores_are_retried_next_run(mock_arxiv, mock_score, *_):
    mock_arxiv.return_value = ([_paper()], {"cs.CV": CategoryStats(1, True)})
    mock_score.side_effect = lambda papers, interests, **kw: ScoringOutcome(scored=[], failed=list(papers))
    with tempfile.TemporaryDirectory() as tmpdir:
        first = _run(tmpdir)
        assert first.score_failed == 1 and first.is_failed      # LLM scored nothing
        mock_score.side_effect = _score_all
        second = _run(tmpdir)                                   # same paper deduped, but retried
        assert second.retry_count == 1 and second.selected_count == 1


@patch("main.fetch_arxiv")
def test_report_warns_when_cutoff_clamped(mock_arxiv):
    mock_arxiv.return_value = ([], {"cs.CV": CategoryStats(0, True)})
    with tempfile.TemporaryDirectory() as tmpdir:
        save_run_state(os.path.join(tmpdir, "state.json"),
                       datetime(2026, 1, 1, tzinfo=timezone.utc), {})
        report = _run(tmpdir)
        assert any("14" in w for w in report.warnings)


@patch("main.send_daily_bot_message", return_value=True)
@patch("main.fetch_arxiv")
def test_alert_bot_sent_even_with_zero_papers(mock_arxiv, mock_bot):
    mock_arxiv.return_value = ([], {"cs.CV": CategoryStats(0, True)})
    with tempfile.TemporaryDirectory() as tmpdir:
        config = _config(tmpdir)
        config["output"]["feishu_bot"]["enabled"] = True
        with patch.dict(os.environ, {config["output"]["feishu_bot"]["webhook_url_env"]: "http://hook"}):
            run_pipeline(config, data_dir=os.path.join(tmpdir, "d"), db_path=os.path.join(tmpdir, "p.db"),
                         site_dir=os.path.join(tmpdir, "s"), state_path=os.path.join(tmpdir, "st.json"))
        kwargs = mock_bot.call_args.kwargs
        assert kwargs["is_alert"] is True and kwargs["paper_count"] == 0


@patch("main.send_daily_bot_message")
@patch("main.fetch_arxiv")
def test_no_publish_skips_bot(mock_arxiv, mock_bot):
    mock_arxiv.return_value = ([], {"cs.CV": CategoryStats(0, True)})
    with tempfile.TemporaryDirectory() as tmpdir:
        config = _config(tmpdir)
        config["output"]["feishu_bot"]["enabled"] = True
        run_pipeline(config, data_dir=os.path.join(tmpdir, "d"), db_path=os.path.join(tmpdir, "p.db"),
                     site_dir=os.path.join(tmpdir, "s"), state_path=os.path.join(tmpdir, "st.json"),
                     publish=False)
        mock_bot.assert_not_called()
```

- [ ] **Step 3: Run to verify failure** (`ImportError` for `attach_paper_contexts` etc. in `main`).

- [ ] **Step 4: Implement `main.py`.** Imports:

```python
from datetime import date, datetime, timedelta, timezone

from fetcher.fulltext import attach_paper_contexts
from filter.llm_scoring import score_papers
from filter.llm_summary import summarize_papers
from filter.selection import normalize_categories, select_papers
from run_report import RunReport
from storage.db import PaperDB, STATUS_KEYWORD_REJECTED, STATUS_SCORE_FAILED
from storage.run_state import compute_arxiv_cutoff, load_run_state, save_run_state
```

(`llm_filter` import stays — `cmd_filter` no longer needs it but `conference_pipeline` imports it itself; remove it from `main.py` if unused after this task.)

`stage_fetch(config, since_date, report) -> list[Paper]`: arXiv branch becomes:

```python
            arxiv_papers, cat_stats = fetch_arxiv(categories=unique_categories, since_date=since_date)
            report.source_counts["arXiv"] = len(arxiv_papers)
            for cat, st in cat_stats.items():
                if not st.complete:
                    report.arxiv_complete = False
                    report.warnings.append(f"arXiv {cat} 抓取不完整（{st.note}），本次不推进运行状态")
            all_papers.extend(arxiv_papers)
```

The `except` sets `report.arxiv_complete = False` and adds `report.errors.append(f"arXiv 抓取异常: {e}")`. HF / S2 / RSS branches add `report.source_counts["HF"] = len(hf_papers)` (and `"S2"`, `"RSS"`) only when enabled. Remove the old `max_results`/`lookback_days` arXiv arguments.

`stage_filter(papers, config, db, report, today) -> list[Paper]`:

```python
def stage_filter(
    papers: list[Paper], config: dict, db: PaperDB, report: RunReport, today: str,
) -> list[Paper]:
    fcfg = config["filter"]
    interests = config["interests"]
    names = [i["name"] for i in interests]

    kw_results = keyword_filter(papers, interests, threshold=fcfg["keyword_threshold"])
    kw_papers = [paper for paper, _ in kw_results]
    area_of = {paper.id: areas[0] for paper, areas in kw_results if areas}
    kw_ids = {p.id for p in kw_papers}
    db.set_filter_status([p.id for p in papers if p.id not in kw_ids], STATUS_KEYWORD_REJECTED)
    report.keyword_count = len(kw_papers)
    log.info("After keyword filter: %d papers", len(kw_papers))
    if not kw_papers:
        return []

    timeout = config.get("enrichment", {}).get("timeout", 10)
    report.html_count = attach_paper_contexts(kw_papers, timeout=timeout)

    llm_kwargs = {
        "model": fcfg["llm_model"],
        "api_key": os.environ.get("ANTHROPIC_API_KEY", ""),
        "base_url": os.environ.get("ANTHROPIC_BASE_URL") or None,
    }
    outcome = score_papers(kw_papers, interests, batch_size=fcfg["score_batch_size"],
                           area_of=area_of, **llm_kwargs)
    report.other_count = normalize_categories(outcome.scored, names)
    for p in outcome.scored:
        db.update_scores(p)
    db.set_filter_status([p.id for p in outcome.failed], STATUS_SCORE_FAILED)
    report.scored_count, report.score_failed = len(outcome.scored), len(outcome.failed)
    if outcome.failed:
        report.warnings.append(f"{len(outcome.failed)} 篇打分失败，下次运行自动重试")
    if not outcome.scored:
        report.errors.append("LLM 打分全部失败")
        return []

    selection = select_papers(outcome.scored, interest_names=names,
                              threshold=fcfg["llm_relevance_threshold"],
                              min_relevance=fcfg["min_relevance"], max_per_area=fcfg["max_per_area"])
    report.selected_count, report.dropped_by_cap = len(selection.selected), selection.dropped_by_cap
    report.summary_missing = summarize_papers(selection.selected, interests,
                                              batch_size=fcfg["llm_batch_size"], **llm_kwargs)
    if report.summary_missing:
        report.warnings.append(f"{report.summary_missing} 篇未生成中文摘要")
    db.mark_selected([p.id for p in selection.selected], today)
    return selection.selected
```

`stage_enrich`: unchanged (it still writes `update_filter_result` for summaries and the enrichment data).

`stage_output(config, db, scored_papers, data_dir, site_dir, today, report, publish) -> str` (returns `summary_text`): body as today with these changes —
- `paper_dicts = db.get_papers_by_date(today)` (drop `min_score`).
- deploy block only when `publish`, recording `report.deploy = _run_deploy(deploy_cmd)`; when not publishing, `report.deploy = "skipped (--no-publish)"`; if deploy status starts with `failed`, `report.errors.append(f"网页部署失败: {status}")`.
- Feishu doc only when `publish`: `report.feishu_doc = generate_feishu_doc(...)`; `failed` → `report.warnings.append(f"飞书文档失败: {report.feishu_doc}")`.
- Remove the bot block from `stage_output` (moved to `stage_notify`).

```python
def _run_deploy(deploy_cmd: str) -> str:
    try:
        result = subprocess.run(shlex.split(deploy_cmd), capture_output=True, text=True, timeout=120)
    except Exception as e:  # noqa: BLE001
        log.warning("Deploy command failed: %s", e)
        return f"failed: {e}"
    if result.returncode != 0:
        log.warning("Deploy failed (rc=%d): %s", result.returncode, result.stderr[:500])
        return f"failed: rc={result.returncode} {result.stderr.strip()[:200]}"
    return "ok"


def stage_notify(config: dict, report: RunReport, summary_text: str, paper_count: int, today: str):
    bot_cfg = config.get("output", {}).get("feishu_bot", {})
    if not bot_cfg.get("enabled", False):
        return
    webhook_url = os.environ.get(bot_cfg.get("webhook_url_env", ""), "")
    if not webhook_url:
        log.warning("FEISHU_BOT_WEBHOOK_URL not set, skipping bot notification")
        return
    pages_base = bot_cfg.get("pages_base_url", "")
    send_daily_bot_message(
        webhook_url=webhook_url,
        summary_text=summary_text,
        date=today,
        paper_count=paper_count,
        pages_url=f"{pages_base}/{today}.html" if pages_base and paper_count else "",
        status_lines=report.status_lines(),
        is_alert=report.is_failed,
    )
```

`run_pipeline`:

```python
def run_pipeline(
    config: dict,
    data_dir: str | None = None,
    db_path: str | None = None,
    site_dir: str | None = None,
    state_path: str | None = None,
    publish: bool = True,
) -> RunReport:
    resolved = _resolve_paths(config)
    data_dir = data_dir or resolved[0]
    db_path = db_path or resolved[1]
    site_dir = site_dir or resolved[2]
    state_path = state_path or os.path.join(os.path.dirname(db_path), "run_state.json")
    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(site_dir, exist_ok=True)

    today = date.today().isoformat()
    started = datetime.now(timezone.utc)
    report = RunReport(date=today)
    acfg, fcfg = config["sources"]["arxiv"], config["filter"]
    since, clamped = compute_arxiv_cutoff(
        load_run_state(state_path), started, acfg["overlap_days"],
        acfg["lookback_days"], acfg["max_lookback_days"])
    if clamped:
        report.warnings.append(
            f"距上次成功运行超过 {acfg['max_lookback_days']} 天，只回补到 {since.date()}，更早的论文可能缺失")
    log.info("arXiv cutoff: %s", since.isoformat())

    summary_text, selected = "", []
    db = PaperDB(db_path)
    try:
        fetched = stage_fetch(config, since.date().isoformat(), report)
        new_papers = stage_dedup(fetched, db)
        retry = db.get_retry_candidates(
            since_iso=(started - timedelta(days=fcfg["retry_window_days"])).isoformat(),
            before_iso=started.isoformat(), limit=fcfg["retry_limit"])
        new_ids = {p.id for p in new_papers}
        candidates = new_papers + [p for p in retry if p.id not in new_ids]
        report.new_count, report.retry_count = len(candidates), len(candidates) - len(new_papers)

        if candidates:
            selected = stage_filter(candidates, config, db, report, today)
        if selected:
            selected = stage_enrich(selected, config, db)
            summary_text = stage_output(config, db, selected, data_dir, site_dir, today, report, publish)
        else:
            with open(os.path.join(data_dir, f"{today}.json"), "w") as f:
                f.write("[]")

        if report.source_counts.get("arXiv", 0) == 0 and config["sources"]["arxiv"]["enabled"]:
            report.errors.append(f"arXiv 返回 0 篇（窗口自 {since.date()}），可能是 API 异常")
        if report.arxiv_complete and not report.is_failed:
            save_run_state(state_path, started, {
                "sources": report.source_counts, "selected": report.selected_count})
    except Exception as e:  # noqa: BLE001 - surface in the bot message instead of dying silently
        log.exception("Pipeline crashed")
        report.errors.append(f"流程异常中断: {e}")
    finally:
        db.close()

    if publish:
        stage_notify(config, report, summary_text, report.selected_count, today)
    log.info("Pipeline complete: %s", " / ".join(report.status_lines()))
    return report
```

Note: the "LLM 打分全部失败" error blocks the state save via `report.is_failed`, satisfying spec item 3.

CLI (`main()`): add flags on the top-level parser and pass them through for the default command:

```python
    parser.add_argument("--db", default=None, help="SQLite path (default data/papers.db)")
    parser.add_argument("--data-dir", default=None, help="Daily JSON dir (default data/daily)")
    parser.add_argument("--site-dir", default=None, help="HTML output dir (default from config)")
    parser.add_argument("--state-file", default=None, help="Run state JSON (default next to DB)")
    parser.add_argument("--no-publish", action="store_true",
                        help="Skip deploy, Feishu doc and bot (for trial runs)")
    ...
    else:
        report = run_pipeline(config, data_dir=args.data_dir, db_path=args.db,
                              site_dir=args.site_dir, state_path=args.state_file,
                              publish=not args.no_publish)
        sys.exit(1 if report.is_failed else 0)
```

`cmd_fetch` / `cmd_filter` / `cmd_output`: update to the new signatures with a throwaway `RunReport(date=today)`: `cmd_fetch` computes `since` via `compute_arxiv_cutoff(load_run_state(...), now, ...)` and calls `stage_fetch(config, since.date().isoformat(), report)`; `cmd_filter` opens the DB and calls `stage_filter(papers, config, db, report, today)` then `stage_enrich`; `cmd_output` calls `stage_output(..., report, publish=True)` then `stage_notify`.

- [ ] **Step 5: Run the full suite.** `python -m pytest -q -p no:cacheprovider` → all PASS.
- [ ] **Step 6: Commit.** `git commit -am "feat: wire paginated fetch, two-stage scoring, selection, run state and run report into main"`

---

### Task 12: Score display on 0–100, summary tokens, S2 log, re_enrich

**Files:**
- Modify: `output/feishu_output.py:77-85,123,179-185`, `templates/daily.html.j2:128`, `output/summary.py:93-101`, `fetcher/s2_search_fetcher.py:25-35`, `re_enrich.py:23`
- Test: `tests/test_feishu_output.py`, `tests/test_summary.py`, `tests/test_html_output.py`, `tests/test_s2_search_fetcher.py`

**Interfaces:** `output/feishu_output.py` gains `format_score(paper: dict) -> str` returning e.g. `"85（相关 36 · 新颖 24 · 质量 25）"` or `"85"` when sub-scores are absent, `"?"` when no score.

- [ ] **Step 1: Write failing tests.** In `tests/test_feishu_output.py` change `test_build_markdown_score_backgrounds` fixture scores from 9/7 to 90/70 (assertions unchanged) and add:

```python
def test_score_emoji_thresholds_on_100_scale():
    from output.feishu_output import _score_emoji
    assert _score_emoji(92) == "fire"
    assert _score_emoji(85) == "star"
    assert _score_emoji(70) == "page_facing_up"


def test_format_score_with_and_without_subscores():
    from output.feishu_output import format_score
    assert format_score({"relevance_score": 85, "score_relevance": 36, "score_novelty": 24,
                         "score_quality": 25}) == "85（相关 36 · 新颖 24 · 质量 25）"
    assert format_score({"relevance_score": 80.0}) == "80"
    assert format_score({}) == "?"
```

In `tests/test_summary.py` change fixture scores to `85` and `90` and the assertion to:

```python
def test_build_prompt_includes_scores():
    prompt = _build_prompt(SAMPLE_PAPERS, INTERESTS)
    assert "[85分]" in prompt
    assert "[90分]" in prompt
```

Append to `tests/test_summary.py`:

```python
@patch("output.summary.anthropic")
def test_call_llm_uses_8192_tokens_and_warns_on_truncation(mock_anthropic, caplog):
    msg = MagicMock()
    block = MagicMock()
    block.type, block.text = "text", "partial"
    msg.content, msg.stop_reason = [block], "max_tokens"
    mock_anthropic.Anthropic.return_value.messages.create.return_value = msg
    assert _call_llm("p", "m", "k", None) == "partial"
    assert mock_anthropic.Anthropic.return_value.messages.create.call_args.kwargs["max_tokens"] == 8192
    assert "truncated" in caplog.text
```

In `tests/test_html_output.py` add (adjust the call to match the file's existing `generate_daily_page` usage):

```python
def test_daily_page_shows_integer_score_and_subscores(tmp_path):
    from output.html_output import generate_daily_page
    papers = [{"title": "T", "url": "u", "relevance_score": 85.0, "score_relevance": 36,
               "score_novelty": 24, "score_quality": 25, "primary_category": "底层视觉",
               "summary_zh": "s", "tags": "[]", "authors": '["A"]'}]
    generate_daily_page(papers, "2026-10-15", str(tmp_path), "PT")
    html = (tmp_path / "2026-10-15.html").read_text()
    assert ">85<" in html
    assert "相关 36 · 新颖 24 · 质量 25" in html
```

In `tests/test_s2_search_fetcher.py` add a test asserting the circuit-breaker log reports the number of *queries* skipped:

```python
@patch("fetcher.s2_search_fetcher.time.sleep")
@patch("fetcher.s2_search_fetcher._search_one_query", return_value=([], True))
def test_circuit_breaker_logs_remaining_query_count(_q, _sleep, caplog):
    from fetcher.s2_search_fetcher import fetch_s2_search
    fetch_s2_search([f"q{i}" for i in range(10)])
    assert "skipping remaining 7 queries" in caplog.text
```

(If `time` is not imported in that module, drop the sleep patch.)

- [ ] **Step 2: Run to verify failures.**

- [ ] **Step 3: Implement.**

`output/feishu_output.py`:

```python
def _score_emoji(score) -> str:
    if score is None or score == "?":
        return ""
    s = float(score)
    if s >= 90:
        return "fire"
    if s >= 80:
        return "star"
    return "page_facing_up"


def format_score(paper: dict) -> str:
    score = paper.get("relevance_score")
    if score is None:
        return "?"
    total = f"{float(score):g}"
    parts = (paper.get("score_relevance"), paper.get("score_novelty"), paper.get("score_quality"))
    if any(v is None for v in parts):
        return total
    return f"{total}（相关 {parts[0]} · 新颖 {parts[1]} · 质量 {parts[2]}）"
```

Replace the two `评分 {score}` renderings with `评分 {format_score(p)}` / `评分 **{format_score(p)}**`, and the background line with `bg = "light-yellow" if float(p.get("relevance_score") or 0) >= 80 else "light-grey"`.

`templates/daily.html.j2:128`:

```jinja
        <span class="score"{% if p.score_relevance is not none %} title="相关 {{ p.score_relevance }} · 新颖 {{ p.score_novelty }} · 质量 {{ p.score_quality }}"{% endif %}>{{ "%d" | format(p.relevance_score) if p.relevance_score is not none else "-" }}</span>
        {% if p.score_relevance is not none %}<span class="subscores">相关 {{ p.score_relevance }} · 新颖 {{ p.score_novelty }} · 质量 {{ p.score_quality }}</span>{% endif %}
```

Add CSS next to `.score` in the template: `.subscores { color: #888; font-size: 12px; margin-left: 6px; }`. In `output/html_output.py` add `p.setdefault("score_relevance", None)` (and novelty/quality) next to the other `setdefault` calls so old JSON renders.

`output/summary.py:_call_llm`: `max_tokens=8192`, and after the call:

```python
    if message.stop_reason == "max_tokens":
        log.warning("Summary LLM response truncated (hit max_tokens=8192)")
```

(Use the module's existing logger; add `log = logging.getLogger(__name__)` if missing.)

`fetcher/s2_search_fetcher.py`: iterate with `for idx, query in enumerate(queries):` and log `len(queries) - idx` instead of `len(queries) - len(seen)`.

`re_enrich.py:23`: replace the hard-coded model with `llm_model = config["filter"]["llm_model"]` (load config the way the script already does; if it does not load config, add `config = yaml.safe_load(open(os.path.join(os.path.dirname(__file__), "config.yaml")))`).

- [ ] **Step 4: Run the full suite** → all PASS.
- [ ] **Step 5: Commit.** `git commit -am "feat: 0-100 score display with sub-scores; summary token budget; S2 log fix"`

---

### Task 13: Trial run, threshold calibration, docs, rollout

**Files:**
- Modify: `config.yaml` (`llm_relevance_threshold`, possibly `min_relevance`), `CLAUDE.md`

- [ ] **Step 1: Prepare an isolated trial environment** (copies, never the live DB):

```bash
cd /home/work/ssd1/yyq/projects/agent_harness/paper-tracker-wt
mkdir -p /tmp/pt-trial
cp ../paper-tracker/data/papers.db /tmp/pt-trial/papers.db
cp ../paper-tracker/.env .env
```

- [ ] **Step 2: Run the pipeline without publishing.**

```bash
/home/yyq/miniforge3/bin/python main.py --db /tmp/pt-trial/papers.db --data-dir /tmp/pt-trial/daily \
  --site-dir /tmp/pt-trial/site --state-file /tmp/pt-trial/run_state.json --no-publish \
  2>&1 | tee /tmp/pt-trial/run.log; echo "exit=$?"
```

Expected: migration runs on the copy; log shows per-category arXiv counts with `complete=True`, Stage A batches, a selection line, `Pipeline complete: 来源: ...`. Because the copied DB already contains papers up to 10-15, most fetched papers dedup away; to calibrate on a realistic volume, also run once with a fresh empty DB:

```bash
rm -f /tmp/pt-trial/fresh.db
/home/yyq/miniforge3/bin/python main.py --db /tmp/pt-trial/fresh.db --data-dir /tmp/pt-trial/daily-fresh \
  --site-dir /tmp/pt-trial/site-fresh --state-file /tmp/pt-trial/fresh_state.json --no-publish \
  2>&1 | tee /tmp/pt-trial/fresh.log
```

- [ ] **Step 3: Inspect the score distribution.**

```bash
/home/yyq/miniforge3/bin/python - <<'EOF'
import sqlite3, collections
c = sqlite3.connect("file:/tmp/pt-trial/fresh.db?mode=ro", uri=True)
rows = c.execute("select relevance_score, score_relevance, primary_category from papers where filter_status='scored'").fetchall()
print("scored:", len(rows))
hist = collections.Counter(int(r[0]) // 5 * 5 for r in rows)
for b in sorted(hist): print(f"{b:3d}-{b+4:3d} {'#' * hist[b]} {hist[b]}")
for cat, n in collections.Counter(r[2] for r in rows if r[0] >= 60).most_common(): print(cat, n)
print("html coverage:", open('/tmp/pt-trial/fresh.log').read().count('have full text'))
EOF
grep -E "arXiv cs|Stage A|Selection|full text|truncated|missing" /tmp/pt-trial/fresh.log
```

Record in the commit body: total scored, histogram, HTML coverage, per-area counts at 60/65/70, wall-clock per stage, any truncation warnings.

- [ ] **Step 4: Choose the threshold with the user.** Present the histogram and per-area counts to the user and ask which cut-off they want (the goal stated in the session: roughly what they can read, ≤50/area is enforced anyway). Set `llm_relevance_threshold` (and `min_relevance` if the histogram shows off-topic papers passing) in `config.yaml` to the agreed values. Also spot-check 5 papers around the threshold with the user (title + sub-scores) to confirm v2.6-pro's strictness is acceptable.

- [ ] **Step 5: Update `CLAUDE.md`** "Architecture" section: replace stages 1–5 with: Fetch (arXiv paginated to `run_state` cutoff − 4 days) → Dedup (`filter_status=pending`, retry queue) → Keyword filter → arXiv HTML context (affiliations, conclusion, code) → Stage A sub-score scoring (0–100) → Selection (category normalization, threshold, 50/area) → Stage B summaries → Author enrichment → Output + bot status report. Replace "If LLM filter fails, pipeline falls back to keyword-only results" with "If LLM scoring fails, papers stay `pending`/`score_failed` and are retried next run; the bot message reports the failure."

- [ ] **Step 6: Full suite + commit.**

```bash
/home/yyq/miniforge3/bin/python -m pytest -q -p no:cacheprovider
git commit -am "chore: calibrate 0-100 threshold from trial run; update CLAUDE.md"
```

- [ ] **Step 7: Rollout (only after the user approves; check `crontab -l` and avoid the 18:00–19:30 window on cron days).**

```bash
cd /home/work/ssd1/yyq/projects/agent_harness/paper-tracker
cp data/papers.db data/papers.db.bak-$(date +%F)          # migration is one-way
git merge --ff-only feat/full-fetch-subscores               # or the user's preferred merge
/home/yyq/miniforge3/bin/python -m pytest -q -p no:cacheprovider
```

Do not push; `deploy.sh` pushes `main` on the next cron run (which also deploys the site). After the next cron run, check `data/cron.log` for `Pipeline complete:` status line, `exit` code, the bot message in the group, and that `data/run_state.json` exists.

---

## Self-review notes

- Spec coverage: items 1–2 → Task 4; 3 → Tasks 3, 11; 4 → Tasks 2, 11; 5 → Tasks 2, 7; 6 → Tasks 5, 7; 7 → Tasks 7, 9; 8 → Task 8; 9–10 → Task 2; 11 → Tasks 11, 13; 12 → Tasks 10, 11; 13 → Task 9.
- Interface names cross-checked: `fetch_arxiv(categories, since_date, ...) -> (papers, stats)`, `CategoryStats`, `attach_paper_contexts`, `score_papers(... batch_size, area_of) -> ScoringOutcome`, `normalize_categories`, `select_papers(... threshold, min_relevance, max_per_area) -> SelectionResult`, `summarize_papers(...) -> int`, `RunReport` (incl. `other_count`, `arxiv_complete`), `get_retry_candidates(since_iso, before_iso, limit)`, `mark_selected(ids, selected_on)`, `send_daily_bot_message(..., status_lines, is_alert)`, `generate_feishu_doc(...) -> str`.
