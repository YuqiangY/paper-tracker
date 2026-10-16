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
