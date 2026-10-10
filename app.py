"""社区康养知识库问答 Demo 的 Streamlit 界面。

启动方式：
    streamlit run app.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from healthcare_rag.config import DEFAULT_KNOWLEDGE_DIR, Settings  # noqa: E402
from healthcare_rag.evaluation import evaluate, format_report_markdown, load_eval_set  # noqa: E402
from healthcare_rag.loader import load_documents  # noqa: E402
from healthcare_rag.pipeline import HealthcareRAG  # noqa: E402
from healthcare_rag.safety import DISCLAIMER  # noqa: E402

st.set_page_config(page_title="社区康养知识库问答 Demo", page_icon="🏥", layout="wide")

EXAMPLES = [
    "老年人每天吃多少盐比较合适？",
    "老人摔倒了应该马上扶起来吗？",
    "家里老人同时吃很多种药，要注意什么？",
    "老人晚上总是睡不好，有什么改善睡眠的方法？",
]


@st.cache_resource(show_spinner="正在加载知识库并构建检索索引…")
def get_pipeline(
    knowledge_dir: str,
    top_k: int,
    min_score: float,
    chunk_size: int,
    chunk_overlap: int,
    use_llm: bool,
) -> HealthcareRAG:
    settings = Settings.from_env(
        knowledge_dir=Path(knowledge_dir),
        top_k=top_k,
        min_score=min_score,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        use_llm=use_llm,
    )
    return HealthcareRAG(settings)


@st.cache_data(show_spinner=False)
def get_documents(knowledge_dir: str):
    documents = load_documents(knowledge_dir)
    return [
        {"title": d.title, "source": d.source, "sections": len(d.sections), "content": d.content}
        for d in documents
    ]


def render_citations(citations) -> None:
    if not citations:
        return
    st.markdown("**引用来源**")
    for cite in citations:
        with st.expander(f"[{cite['index']}] {cite['title']} · {cite['section']}（相似度 {cite['score']:.3f}）"):
            st.markdown(f"> {cite['snippet']}…")
            st.caption(f"来源文件：data/knowledge/{cite['source']}")


# ----------------------------------------------------------------- 侧边栏参数
base_settings = Settings.from_env()

with st.sidebar:
    st.title("🏥 康养问答")
    st.caption("社区康养知识库 RAG Demo")
    st.divider()
    st.subheader("参数设置")
    top_k = st.slider("检索片段数 Top-K", 1, 8, base_settings.top_k)
    min_score = st.slider("拒答阈值", 0.0, 0.6, float(base_settings.min_score), 0.01)
    chunk_size = st.slider("分块长度（字）", 150, 600, base_settings.chunk_size, 50)
    chunk_overlap = st.slider("分块重叠（字）", 0, 150, base_settings.chunk_overlap, 10)
    llm_enabled = st.checkbox(
        "启用大模型生成",
        value=base_settings.llm_ready,
        help="在 .env 中配置 LLM_API_KEY / LLM_BASE_URL / LLM_MODEL 后可用；未配置时使用抽取式回答。",
        disabled=not base_settings.llm_api_key,
    )
    if not base_settings.llm_api_key:
        st.caption("未检测到 `LLM_API_KEY`，将使用内置的抽取式回答（无需联网）。")
    st.divider()

pipeline = get_pipeline(
    str(base_settings.knowledge_dir),
    top_k,
    min_score,
    chunk_size,
    chunk_overlap,
    bool(llm_enabled),
)

with st.sidebar:
    st.subheader("知识库状态")
    stats = pipeline.stats
    st.metric("文档数", stats["文档数"])
    st.metric("片段数", stats["片段数"])
    st.caption(f"回答模式：{stats['回答模式']}")
    st.caption(f"知识库目录：`{base_settings.knowledge_dir}`")

st.title("社区康养知识库问答 Demo")
st.caption(
    "基于本地康养知识文档的检索增强问答（RAG）：BM25 + TF-IDF 混合检索 → 引用溯源 → 抽取式 / 大模型生成。"
)

tab_qa, tab_kb, tab_eval, tab_about = st.tabs(["💬 问答演示", "📚 知识库", "📊 检索评估", "ℹ️ 关于"])

# --------------------------------------------------------------------- 问答页
with tab_qa:
    st.markdown("#### 试试这些问题")
    cols = st.columns(len(EXAMPLES))
    for col, example in zip(cols, EXAMPLES):
        if col.button(example, use_container_width=True):
            st.session_state["question_input"] = example

    if "history" not in st.session_state:
        st.session_state.history = []

    with st.form("ask_form"):
        question = st.text_input(
            "请输入你的问题",
            key="question_input",
            placeholder="例如：老年人每天吃多少盐比较合适？",
        )
        col_submit, col_clear = st.columns([1, 5])
        submitted = col_submit.form_submit_button("提问", type="primary")
        cleared = col_clear.form_submit_button("清空历史")

    if cleared:
        st.session_state.history = []
        st.rerun()

    if submitted and question.strip():
        with st.spinner("检索知识库并生成回答…"):
            result = pipeline.ask(question)
        st.session_state.history.insert(0, result)

    for result in st.session_state.history:
        st.divider()
        st.markdown(f"**🙋 提问：** {result.question}")
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("耗时", f"{result.latency_ms:.0f} ms")
        col2.metric("命中片段", len(result.retrieved))
        col3.metric("回答模式", "拒答" if result.refused else result.mode)
        col4.metric("安全检查", result.safety_level or "通过")
        st.markdown(result.answer)
        render_citations(result.citations)
        if result.note:
            st.caption(f"备注：{result.note}")
        if result.retrieved:
            with st.expander("🔍 检索明细（分数 / 关键词）"):
                st.dataframe(
                    [
                        {
                            "排名": index,
                            "文档": hit["title"],
                            "小节": hit["section"],
                            "融合分": hit["score"],
                            "BM25": hit["bm25"],
                            "TF-IDF": hit["tfidf"],
                            "片段": hit["text"][:60] + "…",
                        }
                        for index, hit in enumerate(result.retrieved, start=1)
                    ],
                    use_container_width=True,
                    hide_index=True,
                )

# ------------------------------------------------------------------- 知识库页
with tab_kb:
    st.markdown("#### 知识库文档")
    st.caption(f"共 {len(get_documents(str(base_settings.knowledge_dir)))} 篇，全部放在 `data/knowledge/` 目录下，可直接新增 Markdown 文件扩充知识。")
    for doc in get_documents(str(base_settings.knowledge_dir)):
        with st.expander(f"📄 {doc['title']}（{doc['sections']} 个小节 · {doc['source']}）"):
            st.markdown(doc["content"])

# ------------------------------------------------------------------- 评估页
with tab_eval:
    st.markdown("#### 检索 / 问答离线评估")
    st.caption(
        "评估集位于 `data/eval/questions.jsonl`，包含知识库内问题与域外问题（应当拒答）。"
        "指标：Hit@1、Hit@K、MRR、关键词覆盖率、拒答准确率。"
    )
    eval_k = st.slider("评估使用的 K", 1, 5, 3, key="eval_k")
    if st.button("▶️ 运行评估（使用当前侧边栏参数）", type="primary"):
        items = load_eval_set()
        progress_bar = st.progress(0.0, text="评估中…")

        def progress(index: int, total: int, item) -> None:
            progress_bar.progress(index / total, text=f"({index}/{total}) {item.question}")

        report = evaluate(pipeline, items, k=eval_k, progress=progress)
        progress_bar.empty()
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Hit@1", f"{report.hit1:.3f}")
        col2.metric(f"Hit@{report.k}", f"{report.hitk:.3f}")
        col3.metric("MRR", f"{report.mrr:.3f}")
        col4.metric("拒答准确率", f"{report.refusal_accuracy:.3f}")
        st.metric("平均耗时", f"{report.avg_latency_ms:.0f} ms")
        st.dataframe(report.details, use_container_width=True, hide_index=True)
        st.download_button(
            "⬇️ 下载 Markdown 报告",
            data=format_report_markdown(report, pipeline.settings).encode("utf-8"),
            file_name="eval_report.md",
            mime="text/markdown",
        )

# --------------------------------------------------------------------- 关于页
with tab_about:
    st.markdown(
        f"""
#### 这是什么？
一个面向**社区康养 / 老年健康**场景的知识库问答 Demo，
完整实现了 RAG（Retrieval-Augmented Generation）的主要环节，
不依赖任何付费服务也能离线运行。

#### 技术链路
1. **文档加载**：读取 `data/knowledge/` 下的 Markdown，解析标题与小节；
2. **中文分块**：按标点切句 → 按长度打包 → 相邻片段保留重叠；
3. **混合检索**：BM25（关键词稀疏检索）+ TF-IDF（词级 + 字级 2-3 gram）；
4. **引用溯源**：每条回答都带来源文档、小节与相似度分数；
5. **安全兜底**：紧急症状关键词 → 直接提示拨打 120；知识库无依据 → 拒答；
6. **答案生成**：默认抽取式回答；配置大模型后自动切换为生成式回答；
7. **离线评估**：Hit@1 / Hit@K / MRR / 拒答准确率，见评估页。

#### 免责声明
> {DISCLAIMER}
"""
    )