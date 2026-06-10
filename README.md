# 企业知识库 RAG Agent 问答系统

这是一个适合 AI Agent / RAG / 大模型应用开发方向展示的本地 Demo 项目。系统基于 FastAPI 提供后端接口，支持文档上传、切分、向量化、Chroma 检索、RAG 问答、Dynamic Prompt、HyDE、RAGAS 评估和原生 Web 控制台。

项目不虚构真实公司经历、用户量或生产数据，定位是“可运行的 Demo 原型”和“本地知识库问答系统”。

## 技术栈

- Python、FastAPI、Pydantic、Uvicorn
- LangChain、LangGraph、Dynamic Prompt、Tool
- Qwen LLM、Qwen `text-embedding-v4`
- DashScope OpenAI-compatible API
- MinerU 官方 API Token
- Chroma 向量库、SQLite
- RAGAS、DeepSeek Judge Model
- 原生 HTML / CSS / JavaScript 前端
- uv 环境管理

## 核心功能

- 上传 `txt`、`md` 文档，复杂文档通过 MinerU API 解析。
- 支持 MinerU 可处理的 `pdf`、`doc/docx`、`ppt/pptx`、图片、HTML 等文件。
- 使用 `MarkdownHeaderTextSplitter` 保留 Markdown 标题结构，再对长文本做二次切分。
- 使用 Qwen Embedding 生成向量并写入 Chroma。
- 使用 Chroma retriever 做 Top-K 检索，并返回 chunk、metadata、score。
- 支持普通 RAG 问答和 SSE 流式问答。
- 支持基于 `thread_id` 的简单对话历史。
- 支持 Dynamic Prompt：自动查询重写、HyDE 虚构答案、检索上下文注入 system prompt。
- 支持 RAGAS 评估，使用 DeepSeek 作为 Judge Model，并缓存 response / contexts 避免重复消耗 token。
- 内置 Web 控制台，可完成上传、问答、检索调试和评估操作。

## 项目结构

```text
app/
  api/routes.py                 FastAPI 接口
  agent/graph.py                LangGraph 工具路由
  agent/dynamic_prompt_agent.py Dynamic Prompt + Query Rewrite + HyDE
  agent/tools.py                Agent 工具
  core/config.py                环境变量和路径配置
  rag/embeddings.py             Qwen / OpenAI / local Embedding
  rag/llm.py                    Prompt 和模型调用
  rag/text_splitter.py          Markdown 和普通文本切分
  rag/vector_store.py           Chroma 向量库封装
  services/document_service.py  文档上传、解析、切分、入库
  services/evaluation_service.py RAGAS 评估服务
  services/mineru_client.py     MinerU API Token 客户端
  services/rag_service.py       普通 RAG 问答流程
  services/vector_store_service.py 向量库状态、调试和清理服务
  static/                       原生 Web 控制台
  storage/database.py           SQLite 文档和对话历史
eval_data/
  ragas_eval_dataset.json       RAGAS 评估数据集
sample_docs/
  company_handbook.md           示例知识库文档
scripts/
  smoke_test.py                 冒烟测试
  evaluate_ragas.py             命令行 RAGAS 评估
```

## 环境安装

进入项目目录：

```powershell
cd D:\pycharm项目\RAG
```

创建并同步 uv 环境：

```powershell
uv venv .venv --python 3.12
uv sync
```

日常新增依赖请使用：

```powershell
uv add 包名
```

PyCharm 解释器选择：

```text
D:\pycharm项目\RAG\.venv\Scripts\python.exe
```

## 环境变量

复制示例配置：

```powershell
copy .env.example .env
```

核心配置示例：

```env
EMBEDDING_PROVIDER="qwen"
CHAT_PROVIDER="qwen"

QWEN_API_KEY="你的 DashScope API Key"
QWEN_BASE_URL="https://dashscope.aliyuncs.com/compatible-mode/v1"
QWEN_CHAT_MODEL="qwen-plus"
QWEN_EMBEDDING_MODEL="text-embedding-v4"
QWEN_EMBEDDING_DIMENSIONS=1024

PDF_PARSER="mineru_api"
MINERU_API_BASE_URL="https://mineru.net/api/v4"
MINERU_API_TOKEN="你的 MinerU API Token"
MINERU_MODEL_VERSION="vlm"
MINERU_LANGUAGE="ch"

DEEPSEEK_API_KEY="你的 DeepSeek API Key"
DEEPSEEK_BASE_URL="https://api.deepseek.com"
DEEPSEEK_CHAT_MODEL="deepseek-chat"

CHUNK_SIZE=700
CHUNK_OVERLAP=120
TOP_K=4
EMBEDDING_BATCH_SIZE=10
```

## 启动项目

推荐使用：

```powershell
uv run uvicorn app.main:app --reload
```

Web 控制台：

```text
http://127.0.0.1:8000/
```

Swagger API 文档：

```text
http://127.0.0.1:8000/docs
```

## Web 控制台

Web 控制台是原生 HTML/CSS/JS 页面，不需要 Node 或前端构建工具。功能包括：

- 文档上传、文档列表、系统状态查看
- 普通 RAG、流式 RAG、Dynamic RAG 问答
- 查看向量库中的 chunk
- 触发 RAGAS 样本生成和评估
- 展示 RAGAS 缓存状态和指标结果

## 常用接口

### 上传文档

```http
POST /api/documents/upload
Content-Type: multipart/form-data
```

PowerShell 示例：

```powershell
curl.exe -X POST "http://127.0.0.1:8000/api/documents/upload" -F "file=@sample_docs/company_handbook.md"
```

### 查看文档列表

```http
GET /api/documents
```

### 查看系统状态

```http
GET /api/status
```

### 查看向量库 chunk

```http
GET /api/vector-store/chunks?limit=20&offset=0
```

### 清空向量库

```http
DELETE /api/vector-store/clear?confirm=true&delete_files=true
```

### 普通 RAG 问答

```http
POST /api/rag/ask
Content-Type: application/json
```

```json
{
  "question": "员工报销需要准备哪些材料？",
  "thread_id": "default",
  "top_k": 3
}
```

### 流式 RAG 问答

```http
POST /api/rag/stream
Content-Type: application/json
```

返回 `text/event-stream`，适合前端逐字展示。

### Dynamic RAG

```http
POST /api/agent/dynamic-rag
Content-Type: application/json
```

```json
{
  "input": "员工报销需要准备哪些材料？",
  "thread_id": "default"
}
```

Dynamic RAG 流程：

```text
用户问题
-> Qwen 查询重写
-> Qwen 生成 HyDE 虚构答案
-> Chroma retriever Top-K 检索
-> retrieved chunks 注入 system prompt
-> Qwen LLM 生成回答
```

### RAGAS 评估

只生成样本，不调用 DeepSeek：

```http
POST /api/evaluation/ragas/build-samples
```

完整评估：

```http
POST /api/evaluation/ragas/run
```

请求体：

```json
{
  "dataset_path": "eval_data/ragas_eval_dataset.json",
  "samples_output_path": "eval_outputs/ragas_samples.json",
  "result_output_path": "eval_outputs/ragas_result.json",
  "top_k": 3,
  "max_samples": 1,
  "force_rebuild": false
}
```

字段说明：

- `max_samples=1`：只评估前 1 条，适合先测试。
- `force_rebuild=false`：复用缓存的 response 和 retrieved_contexts，避免重复消耗 token。
- `force_rebuild=true`：重新检索并重新生成回答。

评估指标：

- 忠实度 (Faithfulness)
- 回答相关性 (Answer Relevancy)
- 上下文精度 (Context Precision)
- 上下文实体召回 (Context Entity Recall)
- 噪声敏感度 (Noise Sensitivity)
- 上下文召回 (Context Recall)

输出文件：

```text
eval_outputs/ragas_samples.json
eval_outputs/ragas_result.json
eval_outputs/ragas_result.csv
```

## 测试

冒烟测试：

```powershell
uv run python scripts\smoke_test.py
```

编译检查：

```powershell
uv run python -m compileall app scripts
```

前端 JS 语法检查：

```powershell
node --check app/static/app.js
```

## 简历亮点

- 构建完整 RAG 流程：文档解析、chunk 切分、Embedding、向量检索、Prompt 增强生成。
- 使用 MinerU 官方 API Token 处理 PDF、Office、图片等复杂文档。
- 使用 Chroma retriever 实现 Top-K 语义检索，并返回 chunk score 便于调试。
- 实现 Dynamic Prompt + Query Rewrite + HyDE，提高检索语义匹配能力。
- 使用 RAGAS + DeepSeek 对 RAG 回答和检索效果进行离线评估，并加入样本缓存降低评估成本。
- 提供 FastAPI 接口和原生 Web 控制台，方便演示完整工作流。
