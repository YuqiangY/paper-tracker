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
    assert normalize_category("多模态", NAMES) == "多模态大模型"
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
    assert vision[0] == "v2"
    assert r.dropped_by_cap == {"底层视觉": 2}
    assert "g1" in [p.id for p in r.selected]


def test_select_order_is_area_then_score():
    papers = [_p("a", 70, cat="AI Agent"), _p("b", 90), _p("c", 80)]
    r = select_papers(papers, interest_names=NAMES, threshold=60, min_relevance=24, max_per_area=50)
    assert [p.id for p in r.selected] == ["b", "c", "a"]
