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
