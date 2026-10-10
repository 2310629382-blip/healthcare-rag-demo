"""healthcare-rag-demo 的核心代码包。

社区康养知识库问答 Demo：加载本地知识文档 → 中文分块 → 混合检索
（BM25 + TF-IDF 词级/字级）→ 引用溯源 → 可选大模型生成答案。
"""

__version__ = "0.1.0"

__all__ = ["__version__"]