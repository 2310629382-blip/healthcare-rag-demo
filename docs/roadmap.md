# 路线图（Roadmap）

## v0.1（当前版本）

- [x] Markdown 知识库加载与中文分块
- [x] BM25 + TF-IDF（词级 / 字级）混合检索
- [x] 引用溯源（文档 · 小节 · 相似度 · 原文片段）
- [x] 双重拒答护栏 + 医疗安全护栏 + 免责声明
- [x] 抽取式回答 / 大模型回答（OpenAI 兼容，可回退）
- [x] Streamlit 网页界面 + 命令行入口
- [x] 离线评估（Hit@1 / Hit@K / MRR / 关键词覆盖率 / 拒答准确率）
- [x] pytest 单元测试 + GitHub Actions CI

## v0.2 语义检索与对比实验

- [ ] 接入中文向量模型（BGE-small-zh / text2vec-base-chinese）
- [ ] 混合策略：BM25 与向量分数做 RRF（Reciprocal Rank Fusion）或加权融合
- [ ] 给出「稀疏 / 稠密 / 混合」三组对比实验表格（Hit@1、Hit@K、MRR、耗时）
- [ ] 缓存 embedding，避免重复计算

## v0.3 检索质量提升

- [ ] 加入 Rerank（bge-reranker-base 或 Cross-Encoder）
- [ ] 查询改写：多轮对话下的指代消解、口语化表达规范化
- [ ] 同义词 / 术语词典（可配置），缓解「吃盐 vs 食盐」这类问题
- [ ] 评估集扩充到 100+ 条，加入人工标注

## v0.4 知识库工程化

- [ ] 支持 PDF / Word / 网页导入与增量索引
- [ ] 索引持久化（保存 embedding 与 BM25 统计量，避免每次重建）
- [ ] 文档版本与更新时间展示，回答里带上「资料更新时间」

## v0.5 生成质量与部署

- [ ] 生成质量评估：忠实度（faithfulness）、答案相关性、引用准确率
- [ ] Prompt 版本管理与 A/B 对比
- [ ] Docker 镜像 + Streamlit Community Cloud 在线 Demo
- [ ] 增加截图与演示 GIF，完善 README