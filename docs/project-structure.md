# 项目目录索引

这份文档用于快速定位代码文件。以后找功能时，优先按下面的目录职责查。

## 顶层入口

```text
app/main.py
  FastAPI 应用入口，负责创建 app、注册路由、挂载静态页面、启动时初始化目录和 PostgreSQL。

app/schemas.py
  Pydantic 请求体和响应体定义，例如 AskRequest、AgentRequest、AgentResponse。
```

## API 层

```text
app/api/routes.py
  所有 FastAPI 接口集中定义。

  常用接口：
  /api/documents/upload
  /api/documents/upload-batch
  /api/rag/ask
  /api/rag/stream
  /api/agent/invoke
  /api/agent/dynamic-rag
  /api/evaluation/ragas/run
```

## Service 业务层

```text
app/services/document_service.py
  文档上传、文本提取、切分、写入 Chroma 和 PostgreSQL。

app/services/rag_service.py
  普通 RAG 问答和流式 RAG 问答。

app/services/agent_service.py
  Agent 接口服务层，负责调用 LangGraph workflow，并保存对话历史。

app/services/vector_store_service.py
  向量库状态、chunk 列表、清空知识库。

app/services/evaluation_service.py
  RAGAS 样本构建和评估运行。

app/services/mineru_client.py
  MinerU API 文档解析客户端。
```

## LangGraph 工作流层

```text
app/workflows/agent_router.py
  LangGraph Agent 路由工作流。

  流程：
  route_intent
    -> query_status / query_documents / search_knowledge
    -> END

  重点：
  - LLM structured output 语义路由
  - add_conditional_edges 条件边
  - graph_path 调试路径

app/workflows/dynamic_rag.py
  Dynamic RAG LangGraph 工作流。

  流程：
  rewrite_query
    -> generate_hyde
    -> retrieve_context
    -> generate_answer
    -> END

  重点：
  - Query Rewrite
  - HyDE
  - Chroma 检索
  - 基于检索上下文生成答案
  - graph_path 调试路径

app/workflows/document_batch.py
  多文档上传 Orchestrator-Worker 工作流。

  流程：
  orchestrator
    -> Send(upload_worker) * N
    -> summarize_uploads
    -> END

  重点：
  - 使用 LangGraph Send API 动态分发多个文件
  - 每个 worker 复用 DocumentService.ingest_upload()
  - 支持多 PDF / 多文档批量入库
```

## Tool 层

```text
app/tools/rag_tools.py
  LangChain tool 定义。

  包含：
  search_knowledge_base
  query_system_status
  query_document_list
```

## RAG 基础能力层

```text
app/rag/vector_store.py
  Chroma 向量库封装，负责写入文档、相似度检索、列出 chunk、清空 collection。

app/rag/embeddings.py
  embedding 模型构建。

app/rag/llm.py
  普通 RAG 的 prompt 构建、非流式回答和流式回答。

app/rag/text_splitter.py
  文本和 Markdown 切分逻辑。
```

## 存储层

```text
app/storage/database.py
  PostgreSQL 连接池、初始化和读写。

  表：
  documents
    文档元数据

  messages
    对话历史
```

## 前端页面

```text
app/static/index.html
  原生 Web 控制台页面结构。

app/static/app.js
  前端接口调用和页面交互。

app/static/styles.css
  页面样式。
```

## 兼容目录

```text
app/agent/
  兼容旧导入路径的包装目录。

  新代码不要优先改这里，应该去：
  app/workflows/
  app/tools/
```

## 常见问题怎么找文件

```text
我要看 FastAPI 接口
  -> app/api/routes.py

我要看请求体/响应体
  -> app/schemas.py

我要看普通 RAG 怎么回答
  -> app/services/rag_service.py
  -> app/rag/llm.py

我要看 LangGraph 路由 Agent
  -> app/workflows/agent_router.py

我要看 Dynamic RAG 图流程
  -> app/workflows/dynamic_rag.py

我要看多 PDF 批量上传
  -> app/workflows/document_batch.py
  -> app/api/routes.py 的 /api/documents/upload-batch

我要看工具定义
  -> app/tools/rag_tools.py

我要看 Chroma 检索
  -> app/rag/vector_store.py

我要看 PostgreSQL 存储
  -> app/storage/database.py

我要看前端按钮调用哪个接口
  -> app/static/app.js
```
