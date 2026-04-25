from __future__ import annotations
import json
from dataclasses import dataclass, field, asdict


@dataclass
class AuthorInfo:
    name: str
    affiliation: str | None = None
    h_index: int | None = None
    semantic_scholar_id: str | None = None


@dataclass
class Paper:
    id: str
    title: str
    authors: list[str]
    abstract: str
    url: str
    source: str
    published: str
    categories: list[str]
    pdf_url: str | None = None
    relevance_score: float | None = None
    primary_category: str | None = None
    summary_zh: str | None = None
    why_relevant: str | None = None
    tags: list[str] | None = None
    authors_enriched: list[dict] | None = None
    venue: str | None = None
    citation_count: int | None = None
    tldr: str | None = None
    doi: str | None = None

    @property
    def max_h_index(self) -> int | None:
        if not self.authors_enriched:
            return None
        indices = [a["h_index"] for a in self.authors_enriched if a.get("h_index") is not None]
        return max(indices) if indices else None

    @property
    def first_author_h_index(self) -> int | None:
        if self.authors_enriched and self.authors_enriched[0].get("h_index") is not None:
            return self.authors_enriched[0]["h_index"]
        return None

    @property
    def authors_display(self) -> list[str]:
        if not self.authors_enriched:
            return self.authors
        result = []
        for a in self.authors_enriched:
            parts = [a["name"]]
            if a.get("h_index") is not None:
                parts.append(f'[h={a["h_index"]}]')
            result.append(" ".join(parts))
        return result

    @property
    def affiliations_unique(self) -> list[str]:
        if not self.authors_enriched:
            return []
        seen: set[str] = set()
        result: list[str] = []
        for a in self.authors_enriched:
            aff = a.get("affiliation")
            if aff and aff not in seen:
                seen.add(aff)
                result.append(aff)
        return result


def papers_to_json(papers: list[Paper]) -> str:
    return json.dumps([asdict(p) for p in papers], ensure_ascii=False, indent=2)


def papers_from_json(json_str: str) -> list[Paper]:
    data = json.loads(json_str)
    return [Paper(**item) for item in data]
