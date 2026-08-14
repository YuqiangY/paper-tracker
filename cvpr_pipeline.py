"""Backward-compatible CVPR entry point.

The generic sweep logic now lives in ``conference_pipeline.run_conference``.
This module keeps ``run_cvpr`` working for the existing ``cvpr`` CLI command,
bridging the old ``sources.cvpr`` config to the new ``conferences.cvpr`` block.
"""
from __future__ import annotations
import logging

from conference_pipeline import run_conference

log = logging.getLogger(__name__)


def run_cvpr(config: dict, limit: int | None = None) -> None:
    # Bridge legacy sources.cvpr config into conferences.cvpr if the latter is absent.
    conferences = config.setdefault("conferences", {})
    if "cvpr" not in conferences:
        legacy = config.get("sources", {}).get("cvpr", {})
        year = legacy.get("year", 2026)
        conferences["cvpr"] = {
            "fetcher": "cvpr",
            "year": year,
            "slug": f"cvpr{year}",
            "title": f"CVPR {year} 精选",
            "concurrency": legacy.get("concurrency", 4),
            "timeout": legacy.get("timeout", 15),
        }
    run_conference(config, "cvpr", limit=limit)
