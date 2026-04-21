from __future__ import annotations
import re
from models import Paper


def keyword_filter(
    papers: list[Paper],
    interests: list[dict],
    threshold: int = 1,
) -> list[tuple[Paper, list[str]]]:
    results = []
    for paper in papers:
        text = f"{paper.title} {paper.abstract}".lower()
        matched_areas = []
        total_hits = 0

        for interest in interests:
            hits = 0
            for kw in interest["keywords"]:
                if re.search(re.escape(kw.lower()), text):
                    hits += 1
            if hits > 0:
                matched_areas.append(interest["name"])
                total_hits += hits

        if total_hits >= threshold:
            results.append((paper, matched_areas))

    return results
