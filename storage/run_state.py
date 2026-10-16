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
