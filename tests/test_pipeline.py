"""端到端问答流程测试。"""

from __future__ import annotations


def test_answer_in_domain_question(pipeline):
    result = pipeline.ask("老年人每天吃多少盐比较合适？")
    assert not result.refused
    assert result.answer
    assert result.citations, "回答必须带引用来源"
    assert result.retrieved
    assert result.latency_ms >= 0


def test_refuse_out_of_domain_question(pipeline):
    result = pipeline.ask("帮我规划一条去西藏的自驾路线")
    assert result.refused
    assert result.mode == "refuse"
    assert "知识库" in result.answer


def test_empty_question_is_rejected(pipeline):
    result = pipeline.ask("   ")
    assert result.refused
    assert result.mode == "error"


def test_emergency_question_triggers_safety_hint(pipeline):
    result = pipeline.ask("老人突然胸痛、说话不清怎么办？")
    assert result.safety_level == "emergency"
    assert "120" in result.answer


def test_medication_question_triggers_caution(pipeline):
    result = pipeline.ask("降压药能不能自己减量？")
    assert result.safety_level == "caution"


def test_diet_question_is_not_flagged_as_medication(pipeline):
    """「每天吃多少盐」是饮食问题，不应被误判成用药提醒。"""
    result = pipeline.ask("老年人每天吃多少盐比较合适？")
    assert result.safety_level != "caution"


def test_answer_contains_disclaimer(pipeline):
    result = pipeline.ask("老人晚上睡不好怎么办？")
    assert "不能替代医生面诊" in result.answer


def test_debug_context_contains_citation_markers(pipeline):
    context = pipeline.debug_context("怎么预防老年人跌倒？")
    assert "[1]" in context