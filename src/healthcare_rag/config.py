"""项目配置。

所有参数都可以通过环境变量或项目根目录的 ``.env`` 文件覆盖。
默认配置保证「零依赖外部服务」也能跑通：不调用大模型，使用抽取式回答。
"""

from __future__ import annotations

import os
from dataclasses import dataclass, replace
from pathlib import Path

# 项目根目录 = src/healthcare_rag/config.py 的上三级
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_KNOWLEDGE_DIR = PROJECT_ROOT / "data" / "knowledge"

# 默认使用火山方舟（豆包）的 OpenAI 兼容接口，可通过环境变量换成任意兼容服务
DEFAULT_LLM_BASE_URL = "https://ark.cn-beijing.volces.com/api/v3"
DEFAULT_LLM_MODEL = "doubao-seed-1-6-250615"


def load_dotenv_file(path: Path | None = None) -> None:
    """极简 .env 读取，避免为 Demo 增加额外依赖。

    只做「KEY=VALUE」解析，已存在的环境变量不会被覆盖。
    """
    path = path or PROJECT_ROOT / ".env"
    if not path.exists():
        return
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        if key.startswith("export "):
            key = key[len("export ") :].strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


def _as_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, "").strip() or default)
    except ValueError:
        return default


def _as_float(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, "").strip() or default)
    except ValueError:
        return default


def _as_bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None or not raw.strip():
        return default
    return raw.strip().lower() in {"1", "true", "yes", "y", "on"}


@dataclass
class Settings:
    """一次问答流程需要用到的全部参数。"""

    knowledge_dir: Path = DEFAULT_KNOWLEDGE_DIR
    top_k: int = 4                     # 最终送给生成阶段的片段数
    chunk_size: int = 300              # 每个片段的目标字数
    chunk_overlap: int = 60            # 相邻片段的重叠字数
    min_score: float = 0.22            # 低于该分数视为「知识库没有依据」→ 拒答（经评估集校准）
    min_coverage: float = 0.20         # 命中的查询词占比低于该值时同样拒答（防止靠个别虚词蒙分）
    max_context_chars: int = 2600      # 拼进 prompt 的上下文总长度上限
    bm25_weight: float = 0.5           # 混合检索中 BM25 的权重
    tfidf_weight: float = 0.5          # 混合检索中 TF-IDF 的权重
    llm_base_url: str = ""
    llm_api_key: str = ""
    llm_model: str = ""
    llm_temperature: float = 0.2
    llm_timeout: float = 45.0
    use_llm: bool = False              # 是否允许调用大模型

    @classmethod
    def from_env(cls, dotenv_path: Path | None = None, **overrides) -> "Settings":
        """读取 .env / 环境变量，再用显式传入的参数覆盖。"""
        load_dotenv_file(dotenv_path)
        knowledge_dir = os.environ.get("KNOWLEDGE_DIR", "").strip()
        settings = cls(
            knowledge_dir=Path(knowledge_dir).expanduser() if knowledge_dir else DEFAULT_KNOWLEDGE_DIR,
            top_k=_as_int("TOP_K", 4),
            chunk_size=_as_int("CHUNK_SIZE", 300),
            chunk_overlap=_as_int("CHUNK_OVERLAP", 60),
            min_score=_as_float("MIN_SCORE", 0.22),
            min_coverage=_as_float("MIN_COVERAGE", 0.20),
            max_context_chars=_as_int("MAX_CONTEXT_CHARS", 2600),
            bm25_weight=_as_float("BM25_WEIGHT", 0.5),
            tfidf_weight=_as_float("TFIDF_WEIGHT", 0.5),
            llm_base_url=os.environ.get("LLM_BASE_URL", "").strip() or DEFAULT_LLM_BASE_URL,
            llm_api_key=os.environ.get("LLM_API_KEY", "").strip(),
            llm_model=os.environ.get("LLM_MODEL", "").strip() or DEFAULT_LLM_MODEL,
            llm_temperature=_as_float("LLM_TEMPERATURE", 0.2),
            llm_timeout=_as_float("LLM_TIMEOUT", 45.0),
            use_llm=_as_bool("USE_LLM", False),
        )
        if overrides:
            settings = replace(settings, **{k: v for k, v in overrides.items() if v is not None})
        return settings

    @property
    def llm_ready(self) -> bool:
        """是否真的具备调用大模型的条件。"""
        return bool(self.use_llm and self.llm_base_url and self.llm_api_key and self.llm_model)

    @property
    def llm_mode_text(self) -> str:
        if self.llm_ready:
            return f"大模型生成（{self.llm_model}）"
        if self.use_llm:
            return "大模型未就绪（缺少 LLM_API_KEY），已回退抽取式回答"
        return "抽取式回答（未启用大模型）"

    def describe(self) -> dict:
        return {
            "知识库目录": str(self.knowledge_dir),
            "Top-K": self.top_k,
            "分块长度": self.chunk_size,
            "分块重叠": self.chunk_overlap,
            "相似度阈值": self.min_score,
            "查询词覆盖率阈值": self.min_coverage,
            "混合检索权重": f"BM25 {self.bm25_weight} / TF-IDF {self.tfidf_weight}",
            "回答模式": self.llm_mode_text,
        }