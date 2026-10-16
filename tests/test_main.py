import json
import os
import tempfile
from datetime import date, datetime, timezone
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


def _run(tmpdir, config=None, **kw):
    return run_pipeline(
        config or _config(tmpdir), data_dir=os.path.join(tmpdir, "daily"),
        db_path=os.path.join(tmpdir, "p.db"), site_dir=os.path.join(tmpdir, "site"),
        state_path=os.path.join(tmpdir, "state.json"), **kw)


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
        with open(os.path.join(tmpdir, "daily", f"{date.today().isoformat()}.json")) as f:
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
        assert any("cs.CV" in a for a in report.alerts)
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
        assert first.score_failed == 1 and first.is_failed
        mock_score.side_effect = _score_all
        second = _run(tmpdir)
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
            _run(tmpdir, config=config)
        kwargs = mock_bot.call_args.kwargs
        assert kwargs["is_alert"] is True and kwargs["paper_count"] == 0


@patch("main.send_daily_bot_message")
@patch("main.fetch_arxiv")
def test_no_publish_skips_bot(mock_arxiv, mock_bot):
    mock_arxiv.return_value = ([], {"cs.CV": CategoryStats(0, True)})
    with tempfile.TemporaryDirectory() as tmpdir:
        config = _config(tmpdir)
        config["output"]["feishu_bot"]["enabled"] = True
        _run(tmpdir, config=config, publish=False)
        mock_bot.assert_not_called()


@patch("main.attach_paper_contexts", return_value=0)
@patch("main.summarize_papers", side_effect=_summarize)
@patch("main.score_papers", side_effect=_score_all)
@patch("main.fetch_arxiv")
def test_output_crash_requeues_selected_papers(mock_arxiv, *_):
    mock_arxiv.return_value = ([_paper()], {"cs.CV": CategoryStats(1, True)})
    with tempfile.TemporaryDirectory() as tmpdir:
        with patch("main.stage_output", side_effect=RuntimeError("disk full")):
            first = _run(tmpdir)
        assert first.is_failed
        second = _run(tmpdir)
        assert second.retry_count == 1 and second.selected_count == 1


@patch("main.send_daily_bot_message", return_value=True)
@patch("main.PaperDB", side_effect=OSError("database is locked"))
def test_db_open_failure_still_sends_alert(_db, mock_bot):
    with tempfile.TemporaryDirectory() as tmpdir:
        config = _config(tmpdir)
        config["output"]["feishu_bot"]["enabled"] = True
        with patch.dict(os.environ, {config["output"]["feishu_bot"]["webhook_url_env"]: "http://hook"}):
            report = _run(tmpdir, config=config)
        assert report.is_failed
        assert mock_bot.call_args.kwargs["is_alert"] is True


@patch("main.attach_paper_contexts", return_value=0)
@patch("main.summarize_papers", side_effect=_summarize)
@patch("main.score_papers", side_effect=_score_all)
@patch("main.fetch_arxiv")
def test_clamped_window_hitting_max_pages_still_advances_state(mock_arxiv, *_):
    mock_arxiv.return_value = ([_paper()], {"cs.CV": CategoryStats(1, False, "hit max_pages=60")})
    with tempfile.TemporaryDirectory() as tmpdir:
        state = os.path.join(tmpdir, "state.json")
        save_run_state(state, datetime(2026, 1, 1, tzinfo=timezone.utc), {})
        report = _run(tmpdir)
        assert any("max_pages" in w for w in report.warnings)
        with open(state) as f:
            assert json.load(f)["last_success_utc"] != "2026-01-01T00:00:00+00:00"


@patch("main.attach_paper_contexts", return_value=0)
@patch("main.summarize_papers", side_effect=_summarize)
@patch("main.score_papers", side_effect=_score_all)
@patch("main.fetch_arxiv")
def test_warns_when_retry_queue_is_full(mock_arxiv, *_):
    mock_arxiv.return_value = ([_paper()], {"cs.CV": CategoryStats(1, True)})
    with tempfile.TemporaryDirectory() as tmpdir:
        config = _config(tmpdir)
        config["filter"]["retry_limit"] = 1
        from storage.db import PaperDB
        db = PaperDB(os.path.join(tmpdir, "p.db"))
        for pid in ("2610.00090", "2610.00091"):
            db.insert(_paper(pid))
        db.close()
        report = _run(tmpdir, config=config)
        assert any("重试队列" in w for w in report.warnings)


@patch("main.attach_paper_contexts", return_value=0)
@patch("main.summarize_papers", side_effect=_summarize)
@patch("main.score_papers", side_effect=_score_all)
@patch("main.fetch_s2_search")
@patch("main.fetch_arxiv")
def test_s2_papers_older_than_cutoff_are_dropped(mock_arxiv, mock_s2, *_):
    mock_arxiv.return_value = ([_paper()], {"cs.CV": CategoryStats(1, True)})
    old, fresh, yearonly = _paper("s2old"), _paper("s2new"), _paper("s2year")
    old.published, fresh.published, yearonly.published = "2026-03-14", date.today().isoformat(), "2026"
    mock_s2.return_value = [old, fresh, yearonly]
    with tempfile.TemporaryDirectory() as tmpdir:
        config = _config(tmpdir)
        config["sources"]["s2_search"]["enabled"] = True
        report = _run(tmpdir, config=config)
        assert report.source_counts["S2"] == 1
        assert report.new_count == 2


@patch("main.attach_paper_contexts", return_value=0)
@patch("main.summarize_papers", side_effect=_summarize)
@patch("main.score_papers", side_effect=_score_all)
@patch("main.fetch_arxiv")
def test_incomplete_arxiv_is_alert_but_still_outputs(mock_arxiv, *_):
    mock_arxiv.return_value = ([_paper()], {"cs.CV": CategoryStats(1, False, "error: HTTP Error 429")})
    with tempfile.TemporaryDirectory() as tmpdir:
        report = _run(tmpdir)
        assert report.is_alert and not report.is_failed
        assert report.selected_count == 1


def test_config_threshold_and_html_delay():
    config = load_config("config.yaml")
    assert config["filter"]["llm_relevance_threshold"] == 70
    assert config["enrichment"]["html_delay_s"] == 1.0


@patch("main.attach_paper_contexts", return_value=0)
@patch("main.summarize_papers", side_effect=_summarize)
@patch("main.score_papers", side_effect=_score_all)
@patch("main.fetch_arxiv")
def test_run_date_override_labels_outputs(mock_arxiv, *_):
    mock_arxiv.return_value = ([_paper()], {"cs.CV": CategoryStats(1, True)})
    with tempfile.TemporaryDirectory() as tmpdir:
        report = _run(tmpdir, run_date="2026-10-15")
        assert report.date == "2026-10-15"
        assert os.path.exists(os.path.join(tmpdir, "daily", "2026-10-15.json"))
        assert os.path.exists(os.path.join(tmpdir, "site", "2026-10-15.html"))
        from storage.db import PaperDB
        db = PaperDB(os.path.join(tmpdir, "p.db"))
        assert [r["id"] for r in db.get_papers_by_date("2026-10-15")] == ["2610.00001"]
        db.close()
