# 版本记录

版本标签应在对应功能完成并通过验证后创建。下面按功能边界记录当前候选版本；RAGAS 的具体提升百分比仍必须以真实运行结果为准。

## v0.5.0 企业知识运营 Agent 工作台

- 使用 `model -> ToolNode -> model` 有界循环，让模型按需选择知识库、系统查询和文档运营工具；
- RAG 作为 `search_knowledge_base` Tool，普通交流直接回答，明确的企业资料请求由图执行知识证据校验；
- 增加 Tavily 低可信度联网兜底和工作台开关，关闭时由服务端强制阻断网页调用；
- 写操作使用 LangGraph `interrupt()` 审批，并通过 PostgreSQL checkpoint 支持重启后恢复；
- 增加会话切换与删除、来源隔离、LLM 阶段摘要、工具轨迹和 SSE 恢复展示；
- 默认开启 DeepSeek thinking 信号，但不向前端暴露或持久化原始隐藏思维链。

## v0.4.0 PostgreSQL 持久化

- 使用 SQLAlchemy 同步连接池和 psycopg，将运行时业务数据库从 SQLite 迁移到 PostgreSQL；
- Docker Compose 同时启动 FastAPI 和 PostgreSQL，并分别持久化 PostgreSQL、Chroma 和上传文件；
- 增加旧 `data/app.db` 的 dry-run、幂等迁移和 Windows 路径映射脚本；
- CI 和本地测试使用独立 `rag_test` 数据库，健康检查验证 PostgreSQL 连通性。

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
