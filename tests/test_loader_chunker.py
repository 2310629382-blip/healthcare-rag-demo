"""文档加载与分块测试。"""

from __future__ import annotations

import pytest

from healthcare_rag.chunker import chunk_document, chunk_sentences, split_sentences
from healthcare_rag.loader import KnowledgeDoc, load_documents


def test_load_documents_parses_title_and_sections(settings):
    docs = load_documents(settings.knowledge_dir)
    assert len(docs) >= 5
    for doc in docs:
        assert doc.title
        assert doc.content
        assert doc.sections, f"{doc.source} 应该至少解析出一个小节"
        assert all(section_title for section_title, _ in doc.sections if section_title != "")


def test_load_documents_missing_dir(settings):
    with pytest.raises(FileNotFoundError):
        load_documents(settings.knowledge_dir / "not-exist-dir")


def test_split_sentences_splits_on_chinese_punctuation():
    sentences = split_sentences("第一句。第二句！第三句？")
    assert sentences == ["第一句。", "第二句！", "第三句？"]


def test_split_sentences_hard_splits_long_text():
    sentences = split_sentences("甲" * 250, max_sentence_len=100)
    assert len(sentences) == 3
    assert all(len(s) <= 100 for s in sentences)


def test_chunk_sentences_respects_size_and_overlap():
    sentences = ["句子%d。" % i + "内容" * 10 for i in range(12)]
    chunks = chunk_sentences(sentences, chunk_size=80, overlap=20)
    assert len(chunks) > 1
    for chunk in chunks:
        assert len(chunk) <= 80 + 20  # 允许最后追加的一句略微超出


def test_chunk_sentences_rejects_bad_overlap():
    with pytest.raises(ValueError):
        chunk_sentences(["a。"], chunk_size=50, overlap=50)


def test_chunk_document_keeps_metadata():
    doc = KnowledgeDoc(
        doc_id="demo",
        title="演示文档",
        source="demo.md",
        content="",
        sections=[("小节一", "这是第一句话。这是第二句话。")],
    )
    chunks = chunk_document(doc, chunk_size=20, overlap=5)
    assert chunks
    assert chunks[0].doc_id == "demo"
    assert chunks[0].section == "小节一"
    assert chunks[0].doc_title == "演示文档"
    assert "演示文档" in chunks[0].search_text