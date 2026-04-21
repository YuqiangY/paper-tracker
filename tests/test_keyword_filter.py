from models import Paper
from filter.keyword_filter import keyword_filter


def _paper(title, abstract=""):
    return Paper(
        id="test-1",
        title=title,
        authors=[],
        abstract=abstract,
        url="http://x",
        source="arxiv",
        published="2026-04-17",
        categories=["cs.CV"],
    )


INTERESTS = [
    {
        "name": "底层视觉",
        "keywords": ["super-resolution", "denoising", "deblurring"],
        "arxiv_categories": ["cs.CV"],
    },
    {
        "name": "视频算法",
        "keywords": ["video generation", "video editing"],
        "arxiv_categories": ["cs.CV"],
    },
]


def test_matches_keyword_in_title():
    p = _paper("A New Super-Resolution Method")
    results = keyword_filter([p], INTERESTS, threshold=1)
    assert len(results) == 1
    assert results[0][0].id == "test-1"
    assert "底层视觉" in results[0][1]


def test_matches_keyword_in_abstract():
    p = _paper("Some Title", "We propose a denoising approach.")
    results = keyword_filter([p], INTERESTS, threshold=1)
    assert len(results) == 1


def test_no_match_returns_empty():
    p = _paper("Reinforcement Learning for Games", "We train an RL agent.")
    results = keyword_filter([p], INTERESTS, threshold=1)
    assert len(results) == 0


def test_case_insensitive():
    p = _paper("VIDEO GENERATION with diffusion models")
    results = keyword_filter([p], INTERESTS, threshold=1)
    assert len(results) == 1
    assert "视频算法" in results[0][1]


def test_multiple_interest_matches():
    p = _paper(
        "Video Super-Resolution via Denoising",
        "We combine video generation with super-resolution and denoising.",
    )
    results = keyword_filter([p], INTERESTS, threshold=1)
    assert len(results) == 1
    matched_areas = results[0][1]
    assert "底层视觉" in matched_areas
    assert "视频算法" in matched_areas


def test_threshold_filtering():
    p = _paper("A denoising method")  # only 1 keyword match
    results_t1 = keyword_filter([p], INTERESTS, threshold=1)
    results_t3 = keyword_filter([p], INTERESTS, threshold=3)
    assert len(results_t1) == 1
    assert len(results_t3) == 0
