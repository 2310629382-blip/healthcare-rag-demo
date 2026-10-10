"""检索模块测试。"""

from __future__ import annotations

from healthcare_rag.retrieval import BM25, HybridRetriever, tokenize


def test_tokenize_keeps_single_char_content_words():
    tokens = tokenize("控制盐的摄入量")
    assert "盐" in tokens
    assert "的" not in tokens


def test_tokenize_drops_function_words():
    tokens = tokenize("请问我们应该怎么做呢")
    for word in ("的", "呢", "应该", "做"):
        assert word not in tokens


def test_bm25_scores_are_non_negative():
    bm25 = BM25([["高血压", "食盐"], ["糖尿病", "水果"]])
    scores = bm25.score(["高血压"])
    assert len(scores) == 2
    assert scores[0] > scores[1]
    assert scores[1] >= 0


def test_hybrid_search_finds_expected_document(pipeline):
    hits = pipeline.retriever.search("卫生间安装扶手能防止老人跌倒吗？", top_k=3, min_score=0.0)
    assert hits
    assert "老年人跌倒预防与居家改造" in {hit.chunk.doc_id for hit in hits}
    assert hits[0].score >= hits[-1].score
    assert 0.0 <= hits[0].coverage <= 1.0


def test_hybrid_search_empty_query_returns_nothing(pipeline):
    assert pipeline.retriever.search("   ") == []


def test_out_of_domain_query_scores_below_threshold(pipeline):
    hits = pipeline.retriever.search(
        "帮我推荐几部科幻电影", top_k=3, min_score=pipeline.settings.min_score
    )
    assert hits == []