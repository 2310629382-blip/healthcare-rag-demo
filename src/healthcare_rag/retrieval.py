"""混合检索：BM25（词级稀疏检索）+ TF-IDF 余弦相似度（词级 + 字级）。

为什么这样设计？
- BM25 对「高血压」「跌倒」这类关键词命中很敏感，可解释、可调试；
- 中文没有天然空格，字级 TF-IDF（2-3 gram）能兜住分词切错的场景；
- 两路分数各自归一化后加权融合，最终落在 0~1，方便用一个阈值做「拒答」判断。
"""

from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass, field

import jieba
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from .chunker import Chunk

# 停用词只保留最常见的一小部分，避免过度过滤导致召回下降
STOPWORDS = {
    "的", "了", "是", "在", "和", "与", "或", "及", "要", "会", "有", "也", "就", "都", "而",
    "对", "于", "把", "被", "让", "给", "从", "到", "并", "但", "如果", "因为", "所以", "可以",
    "应该", "需要", "什么", "怎么", "如何", "哪些", "一下", "请问", "我们", "你们", "他们",
    "这个", "那个", "这些", "那些", "一个", "没有", "不是", "还是", "以及", "进行", "比较",
}
# 单字功能词，避免保留单字后引入噪声
STOPWORDS |= {
    "啊", "吧", "呗", "但", "的", "地", "得", "等", "都", "而", "该", "个", "给", "更", "还",
    "和", "很", "或", "及", "就", "可", "里", "吗", "没", "们", "哪", "那", "呢", "你", "哦",
    "去", "让", "上", "是", "太", "她", "他", "它", "我", "下", "呀", "也", "以", "又", "于",
    "与", "在", "这", "中", "为", "之", "其", "此", "所", "把", "被", "从", "到", "对", "有",
    # 通用动词/程度词：在很小的语料里 IDF 会被高估，从而把域外问题也「检索」出来
    "能", "做", "少", "多", "买", "买到", "推荐", "写", "帮", "哪个", "一份", "好看", "便宜",
    "应该", "需要", "平时", "合适", "自己", "大家", "可能", "一定", "非常", "特别", "好像",
}
_PUNCT = set("，。！？；：、（）()《》【】[]{}<>\"'`~!@#$%^&*-+=|\\/,.?_:; \t\r\n“”‘’—…·")


def tokenize(text: str) -> list[str]:
    """中文分词 + 停用词过滤。

    单字中文（如「盐」「药」「钙」）往往承载关键信息，必须保留；
    单字西文字母噪声较大，直接丢弃。
    """
    tokens: list[str] = []
    for raw in jieba.lcut(text):
        token = raw.strip().lower()
        if not token or token in _PUNCT or token in STOPWORDS:
            continue
        if all(ch in _PUNCT for ch in token):
            continue
        if len(token) == 1 and token.isascii() and not token.isdigit():
            continue
        tokens.append(token)
    return tokens


class BM25:
    """标准 Okapi BM25 实现（不依赖第三方检索库，方便写进简历/答辩讲原理）。"""

    def __init__(self, corpus_tokens: list[list[str]], k1: float = 1.5, b: float = 0.75) -> None:
        self.k1 = k1
        self.b = b
        self.corpus_tokens = corpus_tokens
        self.term_freqs = [Counter(tokens) for tokens in corpus_tokens]
        self.doc_lengths = np.array([max(len(t), 1) for t in corpus_tokens], dtype=float)
        self.avg_length = float(self.doc_lengths.mean()) if len(corpus_tokens) else 1.0

        document_frequency: Counter[str] = Counter()
        for tf in self.term_freqs:
            document_frequency.update(tf.keys())
        total_docs = len(corpus_tokens)
        self.idf = {
            term: math.log(1 + (total_docs - df + 0.5) / (df + 0.5))
            for term, df in document_frequency.items()
        }

    def score(self, query_tokens: list[str]) -> np.ndarray:
        scores = np.zeros(len(self.corpus_tokens), dtype=float)
        for term in query_tokens:
            idf = self.idf.get(term)
            if idf is None:
                continue
            for index, tf in enumerate(self.term_freqs):
                freq = tf.get(term, 0)
                if not freq:
                    continue
                denominator = freq + self.k1 * (
                    1 - self.b + self.b * self.doc_lengths[index] / (self.avg_length or 1.0)
                )
                scores[index] += idf * freq * (self.k1 + 1) / denominator
        return scores


@dataclass
class RetrievedChunk:
    """一条检索结果，携带分数明细，便于在 UI 里展示「为什么命中」。"""

    chunk: Chunk
    score: float                     # 融合后的 0~1 分数
    bm25_score: float
    tfidf_score: float
    matched_terms: list[str] = field(default_factory=list)
    coverage: float = 0.0          # 命中的查询词占全部查询词的比例（用于拒答判断）

    @property
    def location(self) -> str:
        return self.chunk.location

    def as_dict(self) -> dict:
        return {
            "chunk_id": self.chunk.chunk_id,
            "doc_id": self.chunk.doc_id,
            "title": self.chunk.doc_title,
            "section": self.chunk.section,
            "source": self.chunk.source,
            "score": round(self.score, 4),
            "bm25": round(self.bm25_score, 4),
            "tfidf": round(self.tfidf_score, 4),
            "coverage": round(self.coverage, 4),
            "text": self.chunk.text,
        }


class HybridRetriever:
    """BM25 + TF-IDF 混合检索器。"""

    def __init__(
        self,
        chunks: list[Chunk],
        bm25_weight: float = 0.5,
        tfidf_weight: float = 0.5,
        bm25_saturation: float = 4.0,
        char_weight: float = 0.3,
    ) -> None:
        if not chunks:
            raise ValueError("检索器至少需要 1 个片段")
        self.chunks = list(chunks)
        self.bm25_weight = bm25_weight
        self.tfidf_weight = tfidf_weight
        self.bm25_saturation = bm25_saturation
        self.char_weight = char_weight

        texts = [c.search_text for c in self.chunks]
        self.corpus_tokens = [tokenize(t) for t in texts]
        self.bm25 = BM25(self.corpus_tokens)

        # 词级 TF-IDF：使用 jieba 分词
        self.word_vectorizer = TfidfVectorizer(
            tokenizer=tokenize,
            token_pattern=None,
            lowercase=False,
            sublinear_tf=True,
        )
        self.word_matrix = self.word_vectorizer.fit_transform(texts)

        # 字级 TF-IDF：2-3 gram，兜住分词错误与新词
        self.char_vectorizer = TfidfVectorizer(
            analyzer="char",
            ngram_range=(2, 3),
            lowercase=False,
            sublinear_tf=True,
            min_df=1,
        )
        self.char_matrix = self.char_vectorizer.fit_transform(texts)

    def _tfidf_scores(self, query: str) -> np.ndarray:
        word = cosine_similarity(self.word_matrix, self.word_vectorizer.transform([query]))[:, 0]
        char = cosine_similarity(self.char_matrix, self.char_vectorizer.transform([query]))[:, 0]
        return (1 - self.char_weight) * word + self.char_weight * char

    def search(
        self,
        query: str,
        top_k: int = 4,
        min_score: float = 0.0,
        dedupe_by_doc: bool = True,
    ) -> list[RetrievedChunk]:
        """检索最相关的片段。

        dedupe_by_doc：同一篇文档最多保留 `top_k` 中的 2 条，避免答案来源过于单一。
        """
        query = (query or "").strip()
        if not query:
            return []

        query_tokens = tokenize(query)
        bm25_raw = self.bm25.score(query_tokens) if query_tokens else np.zeros(len(self.chunks))
        bm25_norm = bm25_raw / (bm25_raw + self.bm25_saturation) if bm25_raw.size else bm25_raw
        tfidf = self._tfidf_scores(query)
        fused = self.bm25_weight * bm25_norm + self.tfidf_weight * tfidf

        order = np.argsort(-fused)
        results: list[RetrievedChunk] = []
        per_doc: Counter[str] = Counter()
        for index in order:
            score = float(fused[index])
            if score < min_score:
                break
            chunk = self.chunks[int(index)]
            if dedupe_by_doc and per_doc[chunk.doc_id] >= 2:
                continue
            query_terms = set(query_tokens)
            matched = [t for t in self.corpus_tokens[int(index)] if t in query_terms]
            coverage = len(set(matched)) / len(query_terms) if query_terms else 0.0
            results.append(
                RetrievedChunk(
                    chunk=chunk,
                    score=score,
                    bm25_score=float(bm25_raw[index]),
                    tfidf_score=float(tfidf[index]),
                    matched_terms=matched[:8],
                    coverage=coverage,
                )
            )
            per_doc[chunk.doc_id] += 1
            if len(results) >= top_k:
                break
        return results

    def search_all(self, query: str) -> list[RetrievedChunk]:
        """不截断、不设阈值，返回全部片段按分数排序（用于调试与评估）。"""
        return self.search(query, top_k=len(self.chunks), min_score=0.0, dedupe_by_doc=False)