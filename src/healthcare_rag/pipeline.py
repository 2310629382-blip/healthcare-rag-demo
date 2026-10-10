"""把「加载 → 分块 → 检索 → 安全检查 → 生成」串成一条完整链路。"""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

from .chunker import chunk_documents
from .config import Settings
from .generator import build_citations, build_generator, format_context
from .loader import load_documents
from .retrieval import HybridRetriever, RetrievedChunk
from .safety import DISCLAIMER, build_refusal, check_question


@dataclass
class RAGResult:
    question: str
    answer: str
    mode: str
    citations: list[dict] = field(default_factory=list)
    retrieved: list[dict] = field(default_factory=list)
    refused: bool = False
    refuse_reason: str = ""
    safety_level: str = ""
    note: str = ""
    latency_ms: float = 0.0

    def to_dict(self) -> dict:
        return asdict(self)


class HealthcareRAG:
    """社区康养知识库问答引擎。"""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.documents = load_documents(settings.knowledge_dir)
        self.chunks = chunk_documents(
            self.documents,
            chunk_size=settings.chunk_size,
            overlap=settings.chunk_overlap,
        )
        self.retriever = HybridRetriever(
            self.chunks,
            bm25_weight=settings.bm25_weight,
            tfidf_weight=settings.tfidf_weight,
        )
        self.generator = build_generator(settings)

    # ---------------------------------------------------------------- 基础信息
    @property
    def stats(self) -> dict:
        return {
            "文档数": len(self.documents),
            "片段数": len(self.chunks),
            "平均片段字数": round(sum(len(c.text) for c in self.chunks) / len(self.chunks), 1),
            "回答模式": self.settings.llm_mode_text,
        }

    def retrieve(self, question: str) -> list[RetrievedChunk]:
        return self.retriever.search(question, top_k=self.settings.top_k, min_score=self.settings.min_score)

    def _refusal_text(self, question: str, safety, retrieved) -> str:
        """拒答话术：说明拒答原因，避免让用户以为系统「坏了」。"""
        base = build_refusal(question, self.settings.min_score)
        if not retrieved:
            reason = "检索到的最高相关度低于阈值，知识库里没有足够相关的资料。"
        else:
            reason = (
                f"检索到的片段只覆盖了问题中约 {retrieved[0].coverage:.0%} 的关键词，"
                "证据不足。为避免给出不靠谱的建议，这里选择拒答。"
            )
        parts = [safety.message, reason, base] if safety.message else [reason, base]
        return "\n\n".join(part for part in parts if part)

    # ------------------------------------------------------------------ 主流程
    def ask(self, question: str) -> RAGResult:
        started = time.perf_counter()
        question = (question or "").strip()
        if not question:
            return RAGResult(question="", answer="请输入一个具体的问题。", mode="error", refused=True)

        safety = check_question(question)
        retrieved = self.retrieve(question)
        # 拒答护栏二：即使分数过线，如果只命中了问题里的极少数词，也说明缺少依据
        low_coverage = bool(retrieved) and retrieved[0].coverage < self.settings.min_coverage

        if not retrieved or low_coverage:
            return RAGResult(
                question=question,
                answer=self._refusal_text(question, safety, retrieved),
                mode="refuse",
                refused=True,
                refuse_reason=(
                    "低于相似度阈值"
                    if not retrieved
                    else f"查询词覆盖率 {retrieved[0].coverage:.2f} 低于 {self.settings.min_coverage:.2f}"
                ),
                safety_level=safety.level,
                latency_ms=(time.perf_counter() - started) * 1000,
            )

        answer = self.generator.generate(question, retrieved)
        text = answer.text
        if safety.message:
            text = f"{safety.message}\n\n{text}"
        if DISCLAIMER not in text:
            text = f"{text}\n\n> {DISCLAIMER}"

        return RAGResult(
            question=question,
            answer=text,
            mode=answer.mode,
            citations=answer.citations or build_citations(retrieved),
            retrieved=[item.as_dict() for item in retrieved],
            refused=False,
            safety_level=safety.level,
            note=answer.note,
            latency_ms=(time.perf_counter() - started) * 1000,
        )

    # 调试用：把拼给大模型的上下文原样打印出来
    def debug_context(self, question: str) -> str:
        retrieved = self.retrieve(question)
        return format_context(retrieved, self.settings.max_context_chars)


def build_default_pipeline(knowledge_dir: Path | str | None = None, **overrides) -> HealthcareRAG:
    """便捷构造：读取环境变量后构建引擎。"""
    kwargs = dict(overrides)
    if knowledge_dir is not None:
        kwargs["knowledge_dir"] = Path(knowledge_dir)
    return HealthcareRAG(Settings.from_env(**kwargs))