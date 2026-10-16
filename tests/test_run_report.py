from run_report import RunReport


def test_status_lines_funnel_and_outputs():
    r = RunReport(date="2026-10-15")
    r.source_counts = {"arXiv": 812, "HF": 36, "S2": 0}
    r.new_count, r.retry_count, r.keyword_count = 760, 5, 370
    r.scored_count, r.score_failed, r.selected_count = 365, 5, 180
    r.dropped_by_cap = {"AI Agent": 50}
    r.feishu_doc, r.deploy = "ok", "ok"
    lines = r.status_lines()
    assert lines[0] == "来源: arXiv 812 | HF 36 | S2 0"
    assert "新增 760（含重试 5）→ 关键词 370 → 打分 365（失败 5）→ 入选 180" in lines[1]
    assert "截断: AI Agent -50" in lines[1]
    assert lines[2] == "输出: 网页 ok | 飞书文档 ok"
    assert r.is_failed is False


def test_errors_and_warnings_rendered():
    r = RunReport(date="d")
    r.warnings.append("arXiv cs.AI 不完整")
    r.errors.append("arXiv 返回 0 篇")
    lines = r.status_lines()
    assert "⚠️ arXiv cs.AI 不完整" in lines and "❌ arXiv 返回 0 篇" in lines
    assert r.is_failed is True


def test_alert_when_warning_marks_alert():
    r = RunReport(date="d")
    r.alerts.append("arXiv cs.CV 抓取不完整")
    assert r.is_alert and not r.is_failed
    assert "⚠️ arXiv cs.CV 抓取不完整" in r.status_lines()
    r2 = RunReport(date="d")
    r2.errors.append("x")
    assert r2.is_alert and r2.is_failed
