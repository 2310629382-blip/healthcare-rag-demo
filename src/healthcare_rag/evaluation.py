"""离线评估：检索质量 + 拒答能力 + 端到端耗时。

评估集是 data/eval/questions.jsonl，每行一个 JSON：
{"question": "...", "expected_docs": ["文档名"], "should_refuse": false, "keywords": ["..."]}

指标：
- Hit@1 / Hit@K：Top-1 / Top-K 片段里是否包含期望文档；
- MRR：第一个正确文档排名的倒数均值；
- 关键词覆盖率：答案是否覆盖了期望关键词（生成质量的粗粒度代理指标）；
- 拒答准确率：知识库外的问题是否被正确拒答。
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from .pipeline import HealthcareRAG

DEFAULT_EVAL_PATH = Path(__file__).resolve().parents[2] / "data" / "eval" / "questions.jsonl"


@dataclass
class EvalItem:
    question: str
    expected_docs: list[str] = field(default_factory=list)
    should_refuse: bool = False
    keywords: list[str] = field(default_factory=list)


@dataclass
class EvalReport:
    total: int
    answerable: int
    refusal_questions: int
    hit1: float
    hitk: float
    mrr: float
    keyword_coverage: float
    refusal_accuracy: float
    avg_latency_ms: float
    k: int
    details: list[dict] = field(default_factory=list)

    def as_dict(self) -> dict:
        data = self.__dict__.copy()
        data.pop("details")
        return data


def load_eval_set(path: Path | str = DEFAULT_EVAL_PATH) -> list[EvalItem]:
    path = Path(path)
    items: list[EvalItem] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        payload = json.loads(line)
        items.append(
            EvalItem(
                question=payload["question"],
                expected_docs=payload.get("expected_docs", []),
                should_refuse=payload.get("should_refuse", False),
                keywords=payload.get("keywords", []),
            )
        )
    return items


def _empty_report(k: int) -> EvalReport:
    return EvalReport(0, 0, 0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, k)


def evaluate(pipeline: HealthcareRAG, items: list[EvalItem], k: int = 3, progress=None) -> EvalReport:
    if not items:
        return _empty_report(k)

    answerable = [item for item in items if not item.should_refuse]
    refusals = [item for item in items if item.should_refuse]

    hit1 = hitk = 0
    reciprocal_ranks: list[float] = []
    keyword_hits = 0
    refusal_hits = 0
    latencies: list[float] = []
    details: list[dict] = []

    for index, item in enumerate(items, start=1):
        result = pipeline.ask(item.question)
        latencies.append(result.latency_ms)
        retrieved_doc_ids = [hit["doc_id"] for hit in result.retrieved]

        if item.should_refuse:
            ok = result.refused
            refusal_hits += int(ok)
            details.append(
                {
                    "问题": item.question,
                    "类型": "域外问题",
                    "期望": "拒答",
                    "实际": "拒答" if result.refused else "回答了",
                    "是否通过": ok,
                    "耗时(ms)": round(result.latency_ms, 1),
                }
            )
        else:
            ranks = [i for i, doc_id in enumerate(retrieved_doc_ids, start=1) if doc_id in item.expected_docs]
            first_rank = min(ranks) if ranks else None
            hit1 += int(first_rank == 1)
            hitk += int(first_rank is not None and first_rank <= k)
            reciprocal_ranks.append(1.0 / first_rank if first_rank else 0.0)
            keyword_ok = any(keyword in result.answer for keyword in item.keywords) if item.keywords else True
            keyword_hits += int(keyword_ok)
            details.append(
                {
                    "问题": item.question,
                    "类型": "知识库内",
                    "期望": "、".join(item.expected_docs),
                    "实际": retrieved_doc_ids[0] if retrieved_doc_ids else "无",
                    "排名": first_rank or 0,
                    "关键词命中": keyword_ok,
                    "耗时(ms)": round(result.latency_ms, 1),
                }
            )
        if progress:
            progress(index, len(items), item)

    report = EvalReport(
        total=len(items),
        answerable=len(answerable),
        refusal_questions=len(refusals),
        hit1=hit1 / len(answerable) if answerable else 0.0,
        hitk=hitk / len(answerable) if answerable else 0.0,
        mrr=sum(reciprocal_ranks) / len(reciprocal_ranks) if reciprocal_ranks else 0.0,
        keyword_coverage=keyword_hits / len(answerable) if answerable else 0.0,
        refusal_accuracy=refusal_hits / len(refusals) if refusals else 0.0,
        avg_latency_ms=sum(latencies) / len(latencies),
        k=k,
        details=details,
    )
    return report


def format_report_markdown(report: EvalReport, settings=None) -> str:
    lines = [
        "# 检索 / 问答评估报告",
        "",
        f"- 评测问题数：{report.total}（知识库内 {report.answerable}，域外 {report.refusal_questions}）",
        f"- Hit@1：{report.hit1:.3f}",
        f"- Hit@{report.k}：{report.hitk:.3f}",
        f"- MRR：{report.mrr:.3f}",
        f"- 关键词覆盖率：{report.keyword_coverage:.3f}",
        f"- 拒答准确率：{report.refusal_accuracy:.3f}",
        f"- 平均端到端耗时：{report.avg_latency_ms:.0f} ms",
        "",
    ]
    if settings is not None:
        lines += ["## 配置", ""]
        lines += [f"- {key}：{value}" for key, value in settings.describe().items()]
        lines += [""]
    lines += ["## 明细", "", "| # | 问题 | 类型 | 期望 | 实际 | 结果 |", "| --- | --- | --- | --- | --- | --- |"]
    for index, row in enumerate(report.details, start=1):
        result = "通过" if row.get("是否通过", row.get("关键词命中", False)) else "未通过"
        lines.append(
            f"| {index} | {row['问题']} | {row['类型']} | {row['期望']} | {row['实际']} | {result} |"
        )
    lines.append("")
    return "\n".join(lines)