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
