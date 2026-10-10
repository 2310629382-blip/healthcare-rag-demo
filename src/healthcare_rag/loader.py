"""知识文档加载：把 data/knowledge 下的 Markdown / 纯文本读成结构化文档。"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

SUPPORTED_SUFFIXES = (".md", ".markdown", ".txt")


@dataclass
class KnowledgeDoc:
    """一篇知识文档。"""

    doc_id: str
    title: str
    source: str                 # 磁盘上的文件名，用于引用展示
    content: str                # 去掉标题行之后的正文
    sections: list[tuple[str, str]] = field(default_factory=list)  # [(小节标题, 小节正文)]

    @property
    def char_count(self) -> int:
        return len(self.content)


def _parse_markdown(text: str) -> tuple[str, list[tuple[str, str]]]:
    """解析 Markdown：# 是文档标题，## 是小节。"""
    title = ""
    sections: list[tuple[str, str]] = []
    current_title = ""
    buffer: list[str] = []

    def flush() -> None:
        body = "\n".join(buffer).strip()
        if body:
            sections.append((current_title, body))

    for raw_line in text.splitlines():
        line = raw_line.rstrip()
        if line.startswith("## "):
            flush()
            buffer = []
            current_title = line[3:].strip()
        elif line.startswith("# "):
            if not title:
                title = line[2:].strip()
            flush()
            buffer = []
            current_title = ""
        else:
            buffer.append(line)
    flush()

    if not sections:
        sections = [("", text.strip())]
    return title, sections


def load_document(path: Path) -> KnowledgeDoc:
    content = path.read_text(encoding="utf-8", errors="ignore").strip()
    title, sections = _parse_markdown(content)
    return KnowledgeDoc(
        doc_id=path.stem,
        title=title or path.stem,
        source=path.name,
        content=content,
        sections=sections,
    )


def load_documents(directory: Path | str) -> list[KnowledgeDoc]:
    """加载目录下所有受支持的文档，按文件名排序，保证结果可复现。"""
    directory = Path(directory)
    if not directory.exists():
        raise FileNotFoundError(f"知识库目录不存在：{directory}")
    files = sorted(
        (p for p in directory.iterdir() if p.is_file() and p.suffix.lower() in SUPPORTED_SUFFIXES),
        key=lambda p: p.name,
    )
    if not files:
        raise FileNotFoundError(f"知识库目录里没有可供加载的文档：{directory}")
    return [load_document(p) for p in files]