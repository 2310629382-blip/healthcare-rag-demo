"""中文友好的文本分块。

策略：先按标点切句，再按目标长度把句子打包成片段，相邻片段保留一定重叠，
避免答案被切断。每个片段会保留它所属的文档与小节，方便引用溯源。
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from .loader import KnowledgeDoc

_SENTENCE_SPLIT = re.compile(r"(?<=[。！？!?；;])")
_HEADING_RE = re.compile(r"^#{1,6}\s*")


def split_sentences(text: str, max_sentence_len: int = 200) -> list[str]:
    """按中文标点切句；过长的句子再做硬切分。"""
    sentences: list[str] = []
    for block in text.split("\n"):
        block = block.strip()
        if not block:
            continue
        block = _HEADING_RE.sub("", block)
        for piece in _SENTENCE_SPLIT.split(block):
            piece = piece.strip()
            if not piece:
                continue
            if len(piece) <= max_sentence_len:
                sentences.append(piece)
            else:
                for i in range(0, len(piece), max_sentence_len):
                    sentences.append(piece[i : i + max_sentence_len])
    return sentences


def chunk_sentences(sentences: list[str], chunk_size: int, overlap: int) -> list[str]:
    """把句子列表打包成带重叠的片段列表。"""
    if chunk_size <= 0:
        raise ValueError("chunk_size 必须为正整数")
    if overlap < 0 or overlap >= chunk_size:
        raise ValueError("overlap 必须满足 0 <= overlap < chunk_size")

    chunks: list[str] = []
    buffer: list[str] = []
    for sentence in sentences:
        if buffer and sum(len(s) for s in buffer) + len(sentence) > chunk_size:
            chunks.append("".join(buffer).strip())
            tail: list[str] = []
            tail_len = 0
            for previous in reversed(buffer):
                if tail_len + len(previous) > overlap:
                    break
                tail.insert(0, previous)
                tail_len += len(previous)
            buffer = tail
        buffer.append(sentence)
    if buffer:
        chunks.append("".join(buffer).strip())
    return [c for c in chunks if c]


@dataclass
class Chunk:
    """检索的最小单位。"""

    chunk_id: str
    doc_id: str
    doc_title: str
    section: str
    text: str
    source: str
    position: int

    @property
    def search_text(self) -> str:
        """用于建索引的文本：标题 + 小节 + 正文，提升召回。"""
        return f"{self.doc_title} {self.section} {self.text}".strip()

    @property
    def location(self) -> str:
        return f"{self.doc_title} · {self.section}" if self.section else self.doc_title


def chunk_document(doc: KnowledgeDoc, chunk_size: int = 300, overlap: int = 60) -> list[Chunk]:
    chunks: list[Chunk] = []
    for section_title, section_text in doc.sections:
        sentences = split_sentences(section_text, max_sentence_len=max(chunk_size, 50))
        for piece in chunk_sentences(sentences, chunk_size, overlap):
            chunks.append(
                Chunk(
                    chunk_id=f"{doc.doc_id}#{len(chunks)}",
                    doc_id=doc.doc_id,
                    doc_title=doc.title,
                    section=section_title,
                    text=piece,
                    source=doc.source,
                    position=len(chunks),
                )
            )
    return chunks


def chunk_documents(docs: list[KnowledgeDoc], chunk_size: int = 300, overlap: int = 60) -> list[Chunk]:
    chunks: list[Chunk] = []
    for doc in docs:
        start = len(chunks)
        doc_chunks = chunk_document(doc, chunk_size=chunk_size, overlap=overlap)
        for offset, chunk in enumerate(doc_chunks):
            chunk.chunk_id = f"{doc.doc_id}#{offset}"
            chunk.position = start + offset
        chunks.extend(doc_chunks)
    return chunks