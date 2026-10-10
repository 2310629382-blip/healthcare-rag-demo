"""离线评估脚本。

用法：
    python scripts/evaluate.py
    python scripts/evaluate.py --verbose
    python scripts/evaluate.py --output reports/eval_report.md
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from healthcare_rag.config import Settings  # noqa: E402
from healthcare_rag.evaluation import (  # noqa: E402
    DEFAULT_EVAL_PATH,
    evaluate,
    format_report_markdown,
    load_eval_set,
)
from healthcare_rag.pipeline import HealthcareRAG  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="运行检索 / 问答离线评估")
    parser.add_argument("--eval-file", default=str(DEFAULT_EVAL_PATH), help="评估集路径（jsonl）")
    parser.add_argument("--k", type=int, default=3, help="Hit@K 中的 K")
    parser.add_argument("--top-k", type=int, default=None, help="检索片段数")
    parser.add_argument("--min-score", type=float, default=None, help="拒答阈值")
    parser.add_argument("--output", default=None, help="把 Markdown 报告写入指定文件")
    parser.add_argument("--verbose", action="store_true", help="打印每条问题的明细")
    args = parser.parse_args()

    settings = Settings.from_env(top_k=args.top_k, min_score=args.min_score)
    pipeline = HealthcareRAG(settings)
    items = load_eval_set(args.eval_file)

    print(f"加载 {len(items)} 条评估问题，知识库 {pipeline.stats['文档数']} 篇 / "
          f"{pipeline.stats['片段数']} 个片段，开始评估…")

    def progress(index: int, total: int, item) -> None:
        print(f"  [{index}/{total}] {item.question}")

    report = evaluate(pipeline, items, k=args.k, progress=progress if args.verbose else None)
    markdown = format_report_markdown(report, settings)
    print()
    print(markdown if args.verbose else "\n".join(markdown.splitlines()[:10]))

    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(markdown, encoding="utf-8")
        print(f"报告已写入：{output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())