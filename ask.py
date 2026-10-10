"""命令行问答入口。

用法示例：
    python ask.py "老年人每天吃多少盐合适？"
    python ask.py --show-retrieval "卫生间怎么改造防跌倒？"
    python ask.py            # 进入交互模式
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from healthcare_rag.config import Settings  # noqa: E402
from healthcare_rag.pipeline import HealthcareRAG  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="社区康养知识库问答 Demo（命令行版）")
    parser.add_argument("question", nargs="?", help="要提问的问题，留空则进入交互模式")
    parser.add_argument("--top-k", type=int, default=None, help="检索片段数（默认 4）")
    parser.add_argument("--min-score", type=float, default=None, help="拒答阈值（默认 0.16）")
    parser.add_argument("--knowledge-dir", default=None, help="知识库目录，默认 data/knowledge")
    parser.add_argument("--use-llm", action="store_true", help="启用大模型生成（需配置 LLM_API_KEY）")
    parser.add_argument("--show-retrieval", action="store_true", help="打印检索到的片段与分数")
    parser.add_argument("--debug-context", action="store_true", help="打印拼给大模型的上下文")
    parser.add_argument("--json", action="store_true", help="以 JSON 输出结果")
    return parser


def print_result(result, show_retrieval: bool = False) -> None:
    print("=" * 70)
    print(f"问题：{result.question}")
    print(f"模式：{result.mode}    耗时：{result.latency_ms:.0f} ms    "
          f"{'已拒答' if result.refused else '已作答'}")
    if result.safety_level:
        print(f"安全检查：{result.safety_level}")
    print("-" * 70)
    print(result.answer)
    if result.citations:
        print("-" * 70)
        print("引用来源：")
        for cite in result.citations:
            print(f"  [{cite['index']}] {cite['title']} · {cite['section']} "
                  f"(相似度 {cite['score']:.3f}, 文件 {cite['source']})")
    if show_retrieval and result.retrieved:
        print("-" * 70)
        print("检索明细：")
        for index, hit in enumerate(result.retrieved, start=1):
            print(f"  [{index}] score={hit['score']:.4f} bm25={hit['bm25']:.3f} "
                  f"tfidf={hit['tfidf']:.3f} {hit['title']} · {hit['section']}")
            print(f"      {hit['text'][:80]}…")
    print("=" * 70)


def main() -> int:
    args = build_parser().parse_args()
    overrides = {
        "top_k": args.top_k,
        "min_score": args.min_score,
        "use_llm": True if args.use_llm else None,
    }
    if args.knowledge_dir:
        overrides["knowledge_dir"] = Path(args.knowledge_dir)
    settings = Settings.from_env(**overrides)
    pipeline = HealthcareRAG(settings)

    if not args.json:
        print(f"知识库：{settings.knowledge_dir}")
        print(f"索引：{pipeline.stats['文档数']} 篇文档 / {pipeline.stats['片段数']} 个片段")
        print(f"回答模式：{settings.llm_mode_text}")

    questions = [args.question] if args.question else []
    if questions:
        for question in questions:
            if args.debug_context:
                print(pipeline.debug_context(question))
            result = pipeline.ask(question)
            if args.json:
                print(json.dumps(result.to_dict(), ensure_ascii=False, indent=2))
            else:
                print_result(result, show_retrieval=args.show_retrieval)
        return 0

    print("输入问题后回车（输入 q 退出）：")
    while True:
        try:
            question = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if question.lower() in {"q", "quit", "exit"}:
            break
        if not question:
            continue
        if args.debug_context:
            print(pipeline.debug_context(question))
        result = pipeline.ask(question)
        if args.json:
            print(json.dumps(result.to_dict(), ensure_ascii=False, indent=2))
        else:
            print_result(result, show_retrieval=args.show_retrieval)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())