"""答案生成：默认「抽取式」，配置好大模型后切换为「生成式 + 引用」。

- ExtractiveGenerator：从检索到的片段里挑出最相关的句子拼成答案，
  零成本、零依赖、完全可复现，适合作为基线和离线演示。
- LLMGenerator：调用 OpenAI 兼容的 Chat Completions 接口（豆包 / DeepSeek /
  通义 / 本地 Ollama 均可），要求模型「只依据参考资料回答」并标注引用编号；
  出错时自动回退到抽取式，保证 Demo 不会因为网络问题挂掉。
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass, field

import requests

from .config import Settings
from .retrieval import RetrievedChunk, tokenize
from .safety import DISCLAIMER

SYSTEM_PROMPT = """你是「社区康养知识助手」，面向社区老年人及其家属，回答日常保健、慢病管理、饮食运动、居家安全等问题。

必须遵守的规则：
1. 只依据【参考资料】回答，不要使用资料之外的知识，也不要编造；
2. 每个要点后面用 [编号] 标注依据，例如：每天食盐不超过 5 克 [1]；
3. 资料不足以回答时，直接说明「知识库中没有相关内容」，并建议咨询社区医生；
4. 不做诊断、不开药、不给具体剂量；涉及急症（胸痛、意识不清、跌倒后不能活动等）时，第一句就提醒拨打 120；
5. 用简体中文，分点作答，总长度控制在 300 字以内，语气温和、通俗。"""

USER_TEMPLATE = """【参考资料】
{context}

【问题】
{question}

请依据以上资料作答，并在要点后标注引用编号。"""


@dataclass
class Answer:
    text: str
    mode: str                      # "extractive" / "llm" / "extractive-fallback"
    citations: list[dict] = field(default_factory=list)
    latency_ms: float = 0.0
    note: str = ""


def build_citations(retrieved: list[RetrievedChunk], limit: int = 4) -> list[dict]:
    citations: list[dict] = []
    for rank, item in enumerate(retrieved[:limit], start=1):
        citations.append(
            {
                "index": rank,
                "doc_id": item.chunk.doc_id,
                "title": item.chunk.doc_title,
                "section": item.chunk.section,
                "source": item.chunk.source,
                "score": round(item.score, 4),
                "snippet": item.chunk.text[:160],
            }
        )
    return citations


def format_context(retrieved: list[RetrievedChunk], max_chars: int) -> str:
    blocks: list[str] = []
    used = 0
    for rank, item in enumerate(retrieved, start=1):
        block = f"[{rank}] 《{item.chunk.doc_title}》{(' · ' + item.chunk.section) if item.chunk.section else ''}\n{item.chunk.text}"
        if used + len(block) > max_chars and blocks:
            break
        blocks.append(block)
        used += len(block)
    return "\n\n".join(blocks)


def _sentence_candidates(text: str) -> list[str]:
    parts = re.split(r"(?<=[。！？!?；;])|\n", text)
    return [p.strip() for p in parts if len(p.strip()) >= 8]


class ExtractiveGenerator:
    """抽取式回答：挑出与问题最相关的句子，并标注引用。"""

    mode = "extractive"

    def __init__(self, max_sentences: int = 3) -> None:
        self.max_sentences = max_sentences

    def generate(self, question: str, retrieved: list[RetrievedChunk]) -> Answer:
        started = time.perf_counter()
        query_tokens = set(tokenize(question))
        scored: list[tuple[float, int, str]] = []

        for rank, item in enumerate(retrieved):
            best: tuple[float, str] | None = None
            for sentence in _sentence_candidates(item.chunk.text):
                tokens = set(tokenize(sentence))
                overlap = len(query_tokens & tokens)
                density = overlap / (len(tokens) ** 0.5 + 1e-6)
                if best is None or density > best[0]:
                    best = (density, sentence)
            if best and best[0] > 0:
                scored.append((best[0] + item.score, rank, best[1]))

        if not scored:
            # 没有句子级命中时，退化为「摘录最相关片段的首句」
            excerpts = [item.chunk.text[:120] for item in retrieved[:2]]
            text = "根据知识库中的相关内容：\n\n" + "\n\n".join(f"- {e}…" for e in excerpts)
            return Answer(
                text=f"{text}\n\n> {DISCLAIMER}",
                mode=self.mode,
                citations=build_citations(retrieved),
                latency_ms=(time.perf_counter() - started) * 1000,
                note="未命中句子级关键词，返回片段摘录。",
            )

        scored.sort(key=lambda x: (-x[0], x[1]))
        picked = sorted(scored[: self.max_sentences], key=lambda x: x[1])
        lines = [f"- {sentence} [{rank + 1}]" for _, rank, sentence in picked]
        text = "根据知识库中的相关资料，可以这样理解：\n\n" + "\n".join(lines)
        if len(retrieved) > len({rank for _, rank, _ in picked}):
            text += "\n\n> 还有更多相关说明，可展开下方「引用来源」查看。"
        text += f"\n\n> {DISCLAIMER}"
        return Answer(
            text=text,
            mode=self.mode,
            citations=build_citations(retrieved),
            latency_ms=(time.perf_counter() - started) * 1000,
            note="由检索片段中的原句拼接而成，未经过大模型改写。",
        )


class LLMGenerator:
    """调用 OpenAI 兼容接口生成答案。"""

    mode = "llm"

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.fallback = ExtractiveGenerator()

    def _chat(self, messages: list[dict]) -> str:
        url = self.settings.llm_base_url.rstrip("/") + "/chat/completions"
        response = requests.post(
            url,
            headers={
                "Authorization": f"Bearer {self.settings.llm_api_key}",
                "Content-Type": "application/json",
            },
            json={
                "model": self.settings.llm_model,
                "messages": messages,
                "temperature": self.settings.llm_temperature,
                "stream": False,
            },
            timeout=self.settings.llm_timeout,
        )
        response.raise_for_status()
        payload = response.json()
        return payload["choices"][0]["message"]["content"].strip()

    def generate(self, question: str, retrieved: list[RetrievedChunk]) -> Answer:
        started = time.perf_counter()
        context = format_context(retrieved, self.settings.max_context_chars)
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": USER_TEMPLATE.format(context=context, question=question)},
        ]
        try:
            text = self._chat(messages)
        except Exception as exc:  # noqa: BLE001 - Demo 需要兜住所有网络/接口异常
            fallback = self.fallback.generate(question, retrieved)
            fallback.mode = "extractive-fallback"
            fallback.note = f"大模型调用失败（{type(exc).__name__}: {exc}），已自动回退到抽取式回答。"
            return fallback

        return Answer(
            text=f"{text}\n\n> {DISCLAIMER}",
            mode=self.mode,
            citations=build_citations(retrieved),
            latency_ms=(time.perf_counter() - started) * 1000,
            note=f"由 {self.settings.llm_model} 基于检索片段生成。",
        )


def build_generator(settings: Settings):
    if settings.llm_ready:
        return LLMGenerator(settings)
    return ExtractiveGenerator()