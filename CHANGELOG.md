# 版本记录

版本标签应在对应功能完成并通过验证后创建。下面按功能边界记录当前候选版本；RAGAS 的具体提升百分比仍必须以真实运行结果为准。

## v0.3.0 评估、混合检索与部署

- RAGAS baseline / hyde_rewrite 评估链路与 60 条问题集；
- Hit@1、Hit@3、MRR、分类汇总和阶段耗时；
- BM25 + dense 的 RRF 混合检索；
- 单文档删除、重建索引、API 契约测试、结构化请求日志；
- Docker Compose 单机部署、CI 和 MIT License。

## v0.2.0 MinerU 与批量上传闭环

- MinerU 复杂文档解析和大 PDF 自动拆分；
- 批量上传状态、SHA-256 内容去重、部分失败、补偿清理和受控重试。

## v0.1.0 基础 RAG

- Markdown / 文本切分、Qwen Embedding、Chroma 持久化；
- FastAPI 普通 RAG、SSE 流式问答和轻量 LangGraph Agent。
