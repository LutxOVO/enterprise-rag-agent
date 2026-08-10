# 简历项目经历：企业知识库 RAG Agent 问答系统

## 项目名称

企业知识库 RAG Agent 问答系统

## 推荐写法

**企业知识库 RAG Agent 问答系统｜Python / FastAPI / LangChain / LangGraph / MinerU API / Qwen / Chroma / RAGAS**

- 完成一个面向企业内部知识库问答场景的 RAG Demo 原型，支持上传 txt、md 以及 PDF、Office、图片、HTML 等复杂文档；复杂文档通过 MinerU 官方 API Token 解析为 Markdown，并支持超过接口页数限制的 PDF 自动分片解析。
- 基于 `MarkdownHeaderTextSplitter` 对 Markdown 文档按标题层级切分，并对长章节进行递归二次切分，提升 chunk 的结构完整性和上下文可读性。
- 使用 Qwen `text-embedding-v4` 生成文本向量，基于 Chroma 向量库封装 retriever，实现 dense 检索与 BM25 + dense 的 RRF 混合检索，并返回 chunk score、filename、chunk_index 等元数据，便于答案溯源和调试。
- 使用 DeepSeek Chat 作为 LLM，通过 OpenAI-compatible API 接入 LangChain，将检索片段、历史对话和用户问题拼接为 Prompt，实现基于知识库上下文的问答生成。
- 使用 FastAPI 设计文档上传、批量上传、文档列表、单文档删除、重建索引、系统状态、RAG 问答、流式问答、向量库调试和 RAGAS 评估接口，并通过 Pydantic 定义请求和响应结构。
- 面向批量导入实现 SHA-256 内容幂等去重、文件级状态机、部分失败隔离、补偿清理、失败重试和批次历史；将 MinerU、Embedding、Chroma 与 SQLite 的阻塞操作放入受限 worker。
- 实现基于 `thread_id` 的简单多轮对话历史管理，使用 SQLite 保存用户与助手消息，支持不同会话之间的上下文隔离。
- 使用 LangGraph 构建轻量 Agent 路由和工具工作流，将知识库检索、系统状态查询、文档列表查询封装为工具；这是可观测的路由流程，不把它表述成完全自主的生产 Agent。
- 实现显式 Dynamic RAG 工作流，在模型调用前依次完成 Query Rewrite、HyDE、检索、上下文充分性判断和回答生成，并返回 graph_path 便于调试。
- 提供 Server-Sent Events 流式输出接口，使模型回答可以逐步返回，改善大模型应用的交互体验。
- 引入 RAGAS 评估模块，使用 DeepSeek 作为 Judge Model，对 Faithfulness、Answer Relevancy、Context Precision、Context Entity Recall、Noise Sensitivity、Context Recall 进行离线评估；用同一批 60 条问题比较 baseline / hyde_rewrite，并额外记录 Hit@1、Hit@3、MRR、分类结果和阶段耗时。
- 提供 Docker Compose 单机部署，持久化 SQLite、Chroma、上传文件和 MinerU 输出；通过请求 ID、结构化日志和 API 契约测试增强可排查性。
- 实现原生 HTML/CSS/JS Web 控制台，支持文档上传、问答、Dynamic RAG 检索过程查看、向量 chunk 调试和 RAGAS 评估操作。

## 面试讲解思路

1. 先讲业务目标：解决企业文档分散、人工查找效率低的问题，做一个本地知识库问答 Demo。
2. 再讲核心流程：上传文档 -> MinerU 解析复杂文档 -> Markdown 结构化切分 -> Qwen Embedding -> dense/BM25 检索 -> RRF -> Prompt 拼接 -> DeepSeek 回答。
3. 然后讲 Agentic RAG：加入查询重写、HyDE、上下文充分性判断和轻量工具路由。
4. 最后讲工程化：批次状态、幂等去重、补偿清理、删除/重建索引、FastAPI 契约测试、Docker 和 RAGAS A/B 评估。

## 项目边界说明

- 当前项目是本地 Demo 原型，不虚构真实公司业务数据、用户量或线上收益。
- MinerU 使用官方 API Token，不依赖本地部署 MinerU 模型。
- 向量库使用 Chroma 本地持久化，便于学习和调试；后续可扩展为 Milvus、pgvector 等生产级向量数据库。
- RAGAS 评估会调用 DeepSeek，适合离线评估和效果调试，不建议直接放入高频用户请求链路。

## 模块级表达

### 文档处理模块

实现文档上传和预处理能力，支持 txt、md 直接读取，并支持 PDF、Office、图片、HTML 等复杂文档通过 MinerU API 解析为 Markdown；对 Markdown 先按标题层级切分，再对长文本进行二次切分，生成适合向量检索的 chunk。

### 向量检索模块

封装 Qwen Embedding 和 Chroma 向量库，使用 retriever 实现 dense 检索；对精确编号和专有名词增加 BM25，并用 RRF 融合两路排名，返回相关 chunk、score 和元数据，为 RAG 问答提供可追踪的上下文来源。

### RAG 问答模块

实现从用户问题到检索、Prompt 构建、LLM 回答生成的完整链路，并结合 `thread_id` 保存简单对话历史，支持多会话隔离。

### 流式输出模块

基于 FastAPI `StreamingResponse` 实现 SSE 流式接口，使模型回答能够逐步返回，模拟真实大模型产品中的打字机式交互。

### Agent 工具模块

使用 LangGraph 编排轻量 Agent 工作流，根据用户输入路由到知识库检索、系统状态查询、文档列表查询等工具，体现 Agentic RAG 的基础设计思路。

### Dynamic RAG 模块

使用显式 LangGraph Dynamic RAG 工作流在调用 LLM 前完成查询重写、HyDE 生成、向量检索、上下文充分性判断和 system prompt 注入；保留 `dynamic_prompt_agent.py` 作为兼容入口，当前重点是步骤可观测和可测试。

### RAGAS 评估模块

构建包含问题和标准答案的 60 条评估数据集，分别生成 baseline / hyde_rewrite 样本，并使用 DeepSeek 作为 Judge Model 评估六项 RAGAS 指标；同时用章节标注计算 Hit@1、Hit@3、MRR，按问题类型聚合，并通过缓存机制避免重复生成回答和检索上下文。

### Web 控制台模块

使用原生 HTML/CSS/JS 实现本地 Web 控制台，覆盖文档上传、知识库问答、检索调试和 RAGAS 评估流程，便于演示项目完整能力。

## 后续可扩展方向

- 将 MinerU 解析和 RAGAS 评估改造为后台任务，支持任务状态查询。
- 接入 reranker，提高检索结果排序质量。
- 增加用户登录、文档权限和知识库空间隔离。
- 将 SQLite 扩展为 MySQL/PostgreSQL，并使用 ORM 管理业务表。
- 将 Chroma 替换或扩展为 Milvus、pgvector 等生产级向量数据库。
