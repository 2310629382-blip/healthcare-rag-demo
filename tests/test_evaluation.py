"""评估模块测试。"""

from __future__ import annotations

from healthcare_rag.evaluation import (
    EvalItem,
    evaluate,
    format_report_markdown,
    load_eval_set,
)


def test_load_eval_set_has_both_types(settings):
    items = load_eval_set()
    assert len(items) >= 10
    assert any(item.should_refuse for item in items)
    assert any(not item.should_refuse for item in items)


def test_evaluate_reports_metrics_in_range(pipeline):
    items = [
        EvalItem(question="老人每天吃多少盐？", expected_docs=["高血压日常管理"], keywords=["盐"]),
        EvalItem(question="推荐一部科幻电影", should_refuse=True),
    ]
    report = evaluate(pipeline, items, k=3)
    assert report.total == 2
    assert 0.0 <= report.hit1 <= 1.0
    assert 0.0 <= report.mrr <= 1.0
    assert 0.0 <= report.refusal_accuracy <= 1.0
    assert len(report.details) == 2


def test_format_report_markdown_contains_key_metrics(pipeline):
    items = [EvalItem(question="老人每天吃多少盐？", expected_docs=["高血压日常管理"])]
    markdown = format_report_markdown(evaluate(pipeline, items, k=3), pipeline.settings)
    assert "# 检索 / 问答评估报告" in markdown
    assert "Hit@1" in markdown
    assert "拒答准确率" in markdown


def test_evaluate_empty_items(pipeline):
    report = evaluate(pipeline, [], k=3)
    assert report.total == 0
    assert report.hit1 == 0.0