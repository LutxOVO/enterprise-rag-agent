# 企业知识运营 Agent

[简体中文](README.md) | [English](README.en.md)

[![CI](https://github.com/LutxOVO/enterprise-rag-agent/actions/workflows/ci.yml/badge.svg)](https://github.com/LutxOVO/enterprise-rag-agent/actions/workflows/ci.yml)

一个面向 AI 应用、RAG 学习、面试演示和二次开发的可运行项目。

默认入口不是“每次提问都先查向量库”，而是一个按需决策的企业知识运营 Agent：

- 普通闲聊可以直接回答；
- 需要企业资料时调用知识库 RAG 工具；
- 需要公开信息时调用 Tavily 联网搜索；
- 删除、重建索引、失败重试等写操作暂停等待人工审批；
- 工具轨迹、来源、审批状态和对话状态持久化到 PostgreSQL，服务重启后可以继续。

项目同时保留普通 RAG、Dynamic RAG、批量文档入库、RAGAS 评估和原生 Web 控制台，适合从基础 RAG 一直学习到可恢复 Agent 工作流。

> 这是单机学习和演示用的 Demo 原型，不宣称真实企业部署、线上用户量或生产级高可用。Qwen Embedding、DeepSeek、MinerU 和 Tavily 通过云端 API 调用，不下载本地模型。

当前功能边界按 `CHANGELOG.md` 中的 `v0.5.0` 记录维护；评估数字不会在没有真实运行结果时写入 README。

## 项目主线

这个项目主要解决四类问题：

1. **文档如何进入知识库**：文件经过解析、Markdown 切分、Embedding 后写入 Chroma，并在 PostgreSQL 保存文档元数据。
2. **Agent 什么时候调用 RAG**：RAG 被封装为 search_knowledge_base 工具。普通问题不必检索；明确要求企业资料时必须取证，证据不足时不能凭模型记忆编造。
3. **知识库运营如何可恢复**：批量上传有 SHA-256 去重、文件级状态、部分失败、补偿清理和失败重试；删除、重建索引等写操作需要审批。
4. **效果如何验证**：提供 Dense、Hybrid、Query Rewrite、HyDE、RAGAS 和 Hit@K/MRR 评估入口，区分回答质量和工具行为质量。

核心学习链路：

~~~text
文档入库
  -> MinerU / pypdf 解析
  -> Markdown 标题切分 + 递归字符切分
  -> Qwen text-embedding-v4
  -> Chroma 向量库 + PostgreSQL 元数据
  -> Agent 按需选择 RAG / Web / 文档运营工具
  -> 真实来源约束回答或等待审批
  -> PostgreSQL checkpoint 持久化并恢复
~~~

## 架构总览

~~~mermaid
flowchart LR
    USER[用户 / 原生 Web 控制台] --> API[FastAPI]
    API --> AGENT[企业知识运营 Agent]
    AGENT --> MODEL{DeepSeek 判断任务}
    MODEL -->|无需工具| DIRECT[直接回答]
    MODEL -->|需要工具| TOOLS[LangGraph ToolNode]
    TOOLS --> MODEL

    TOOLS -->|search_knowledge_base| RAG[RAG Tool]
    RAG --> MODE{standard / deep}
    MODE -->|standard| HYBRID[Dense + BM25 + RRF]
    MODE -->|deep| ADVANCED[Query Rewrite + HyDE + 上下文判断]
    HYBRID --> CHROMA[(Chroma)]
    ADVANCED --> CHROMA
    CHROMA --> EVIDENCE[context + 来源 chunks]
    EVIDENCE --> MODEL

    MODEL -->|开关允许| WEB[Tavily Web Search]
    RAG --> CONF{知识库可信度}
    CONF -->|不足且允许联网| WEB
    CONF -->|不足且禁止联网| REFUSE[安全拒答]
    WEB --> MODEL

    TOOLS -->|写工具| APPROVAL{interrupt 审批}
    APPROVAL -->|批准后恢复| WRITE[重试 / 重建索引 / 删除]
    WRITE --> TOOLS

    API --> INGEST[文档入库流水线]
    INGEST --> PARSE[MinerU / pypdf]
    PARSE --> SPLIT[Markdown 标题切分 + 递归切分]
    SPLIT --> EMB[Qwen Embedding]
    EMB --> CHROMA
    INGEST --> PG[(PostgreSQL 业务数据)]
    AGENT --> CHECKPOINT[(PostgreSQL LangGraph checkpoint)]
    API --> EVAL[RAGAS + Hit@K + MRR]
~~~

## 技术栈

| 层次 | 技术 |
| --- | --- |
| API 与运行 | Python 3.12、FastAPI、Pydantic、Uvicorn、uv |
| Agent 编排 | LangChain、LangGraph、ToolNode、interrupt、SSE |
| 对话模型 | DeepSeek Chat，OpenAI-compatible API |
| Embedding | Qwen `text-embedding-v4`，阿里云百炼 OpenAI-compatible API |
| 文档解析 | MinerU 官方 API、pypdf 兜底 |
| 向量检索 | Chroma、Dense、BM25、RRF |
| 持久化 | PostgreSQL、SQLAlchemy、psycopg、LangGraph checkpoint |
| 评估 | RAGAS、DeepSeek Judge、Hit@K、MRR |
| 前端与部署 | 原生 HTML/CSS/JavaScript、Docker Compose |

## 项目展示

仓库中的图片用于展示控制台结构和字段，批次、检索结果使用脱敏演示数据，不代表固定准确率。视频展示一次隔离环境中的完整操作链路。

![文档管理与健康状态](docs/assets/rag-console-documents.png)

![批量上传、重复跳过和失败重试](docs/assets/batch-upload-demo.png)

![检索来源与 Hybrid RRF 调试结果](docs/assets/retrieval-sources-demo.png)

演示视频：[rag-demo.webm](docs/assets/rag-demo.webm)。

## 核心能力

### 1. 按需调用工具的 Agent

- 使用 LangGraph 的 model -> ToolNode -> model 有界循环。
- Agent 可以调用知识库、网页、文档、批次和系统状态工具。
- 关闭并行工具调用，默认最多执行 6 次工具，执行完一个工具后再观察结果。
- 不向 Agent 暴露任意 SQL、Shell、任意文件路径或清空知识库工具。
- SSE 轨迹展示 LLM 阶段、工具调用、工具结果、来源和审批状态；不暴露原始隐藏思维链。

### 2. RAG 作为知识检索工具

- standard：使用原始问题做 Hybrid 检索，融合 Dense 和 BM25 的 RRF 排名。
- deep：先 Query Rewrite，再生成 HyDE 假设答案，用它辅助检索，并进行上下文充分性判断。
- 检索结果携带文件名、文档 ID、chunk、分数和内容摘要。
- 对明确的企业资料问题设置证据闸门，没有可用来源时安全拒答。

### 3. 受控联网搜索

- 工作台有联网搜索开关，关闭时服务端拦截 search_web，不会发出 Tavily 请求。
- 开启后 Agent 可以根据问题主动选择 Tavily；知识库返回低可信度结果时也可以自动兜底一次。
- 用户明确要求联网搜索时，可以直接进入网页工具，不必先查知识库。
- 网页结果标记为 source_type=web，必须展示标题和 URL；网页内容只是资料，不是可执行指令。
- Tavily 失败、为空或额度不足时，系统不会把模型记忆当作网页证据。

### 4. 可恢复的审批工作流

- retry_upload_item、reindex_document、delete_document 是写工具。
- 写工具先生成审批信息，再调用 LangGraph interrupt 暂停。
- 用户使用同一个 thread_id 和 approval_id 批准或拒绝。
- PostgreSQL LangGraph checkpoint 保存消息、工具轨迹、来源、审批状态和最后回答。
- FastAPI 重启后可以恢复等待审批的线程；运行中或等待审批的线程不能被删除。

### 5. 工程化文档入库

- 文件采用分块保存并计算 SHA-256，避免一次性读取大文件。
- 默认限制每批 10 个文件、单文件 50 MB、并发 2、失败最多重试 3 次。
- 批次状态包括 running、completed、partial_success 和 failed。
- 文件状态记录 saving、parsing、splitting、embedding、indexed、duplicate、failed 等阶段。
- Chroma 和 PostgreSQL 无法共享事务，跨存储写入失败时执行补偿删除，减少孤儿向量。
- 单文档支持删除和复用原文件重建索引；历史 SQLite 元数据提供一次性迁移脚本。

### 6. 可评估与可观测

- RAGAS 使用 DeepSeek 作为 Judge，缓存生成的回答和上下文，避免重复消耗 Token。
- 检索侧记录 Hit@1、Hit@3、MRR；回答侧记录 Faithfulness、Answer Relevancy 等指标。
- 记录 Query、检索、上下文和 LLM 阶段耗时。
- 每个请求都有 X-Request-ID，便于从 API 响应串联日志。
- Agent 工具行为评估与 RAGAS 生成质量评估分开，不把两类指标混为一谈。

## 快速启动

### 前置条件

- Docker Desktop，使用 Linux containers。
- Qwen Embedding API Key：上传、Embedding 和检索需要。
- DeepSeek API Key：RAG 回答、Agent 和 RAGAS Judge 需要。
- MinerU API Token：解析 PDF、Office、图片等复杂文件时需要。
- Tavily API Key：打开工作台联网搜索时需要。

### 方式一：Docker Compose，推荐

Docker Compose 会启动 FastAPI 和 PostgreSQL。Qwen、DeepSeek、MinerU、Tavily 仍然通过 .env 访问云端服务。

~~~powershell
git clone https://github.com/LutxOVO/enterprise-rag-agent.git
cd enterprise-rag-agent
copy .env.example .env
~~~

编辑 .env，至少填写：

~~~env
QWEN_API_KEY="你的 Qwen Embedding API Key"
DEEPSEEK_API_KEY="你的 DeepSeek API Key"
~~~

然后启动：

~~~powershell
docker compose config --quiet
docker compose up -d --build
docker compose ps
Invoke-RestMethod http://127.0.0.1:8000/api/health
~~~

访问：

~~~text
Web 控制台：http://127.0.0.1:8000/
Swagger：http://127.0.0.1:8000/docs
健康检查：http://127.0.0.1:8000/api/health
~~~

### 方式二：uv 本地运行 + Compose PostgreSQL

这种方式适合修改 Python 代码并在本机调试，PostgreSQL 仍由 Compose 提供：

~~~powershell
uv venv .venv --python 3.12
uv sync --frozen
copy .env.example .env
~~~

本地 .env 的 DATABASE_URL 使用 localhost，然后启动数据库和应用：

~~~powershell
docker compose up -d postgres
.\scripts\run_local.ps1 -Reload
~~~

如果需要换端口：

~~~powershell
.\scripts\run_local.ps1 -Port 8001 -Reload
~~~

### Docker 更新规则

docker compose restart 只会重启旧容器，不会把宿主机新代码复制进镜像。修改 Python、HTML、CSS、JavaScript、Dockerfile 或依赖后：

~~~powershell
docker compose up -d --build --force-recreate rag
~~~

只修改 .env 时无需构建镜像，重新创建应用容器即可：

~~~powershell
docker compose up -d --force-recreate rag
~~~

普通 docker compose down 不会删除 PostgreSQL 命名卷 postgres_data 和宿主机 data/。不要随意执行 docker compose down -v，它会删除 PostgreSQL 数据。

## 配置说明

完整中文注释见 [.env.example](.env.example)。核心变量如下：

| 配置 | 作用 | 是否必需 |
| --- | --- | --- |
| DATABASE_URL | PostgreSQL 业务库和 LangGraph checkpoint | 必需 |
| QWEN_API_KEY | Qwen text-embedding-v4 文本向量化 | 上传、检索和 RAG 必需 |
| DEEPSEEK_API_KEY | 回答、Agent 路由、Dynamic RAG 和 RAGAS Judge | 问答、Agent、评估必需 |
| MINERU_API_TOKEN | MinerU 云端文档解析 | 复杂文档时需要 |
| TAVILY_API_KEY | Agent 联网搜索 | 打开联网开关时需要 |
| RAG_PORT | Docker 宿主机 API 端口 | 默认 8000 |
| POSTGRES_PORT | Docker 宿主机 PostgreSQL 端口 | 默认 5432 |

模型职责明确分离：

- Qwen text-embedding-v4：把文本转换为向量，服务于写入和检索。
- DeepSeek：生成 RAG 回答、选择 Agent 工具、执行 Dynamic RAG 辅助步骤和作为 RAGAS Judge。
- MinerU：把复杂文档解析为 Markdown，不负责向量化和回答。
- Tavily：只负责公开网页搜索。

Qwen 官方兼容接口的 URL 中包含 dashscope 是正常的；项目配置名统一使用 QWEN_*，并不表示运行时还保留另一套 DashScope Embedding 实现。切换 Embedding 模型后，旧向量和新向量不在同一个语义空间，需要清理并重建知识库。

### 数据持久化

~~~text
PostgreSQL 命名卷 postgres_data
  -> 文档元数据、批次、指纹、旧 RAG 对话、LangGraph checkpoint

宿主机 ./data
  -> Chroma、原始上传文件、MinerU 解析结果
~~~

容器内 PostgreSQL 主机名是 postgres；本地 Python 启动时是 localhost。应用保持单进程运行，因为 Chroma、本地文件处理和进程内锁都面向单机 Demo 设计。

### MinerU 大 PDF

MinerU 单次请求有页数限制，项目默认 MINERU_MAX_PAGES_PER_REQUEST=200：

1. 上传后用 pypdf 读取 PDF 页数。
2. 不超过限制时直接调用 MinerU。
3. 超过限制时按最多 200 页拆分临时 PDF。
4. 分片分别解析，按页序合并 Markdown。
5. 进入标题切分、Embedding 和 Chroma 入库。

用户只需要上传一次原文件，临时分片会自动清理；原文件和解析结果会保留，便于失败重试。详细说明见 [docs/mineru-pdf-processing.md](docs/mineru-pdf-processing.md)。

## 运行一次完整流程

### 1. 上传示例文档

启动服务后，在控制台上传 sample_docs/company_handbook.md，或使用 PowerShell：

~~~powershell
curl.exe -X POST "http://127.0.0.1:8000/api/documents/upload" -F "file=@sample_docs/company_handbook.md"
~~~

复杂文件也可以调用批量接口：

~~~powershell
curl.exe -X POST "http://127.0.0.1:8000/api/documents/upload-batch" -F "files=@C:\path\to\your.pdf"
~~~

### 2. 用 Agent 问答

打开 Web 控制台，默认进入企业知识运营 Agent。可以尝试：

~~~text
根据公司手册说明报销需要哪些材料？
列出当前知识库中的文档。
请分析最近一次失败的上传批次。
~~~

第一类问题会调用 search_knowledge_base 并引用来源；文档和批次操作使用只读工具；重试、重建索引和删除会显示审批卡。

### 3. 观察 Agent 轨迹

典型 SSE 顺序如下：

~~~text
run_started
  -> llm_stage：模型分析任务
  -> tool_call：调用知识库 / 文档 / 网页工具
  -> tool_result：工具结果和来源
  -> llm_stage：分析工具结果
  -> answer：最终回答和来源
  -> done
~~~

轨迹只展示安全的阶段摘要、工具名、参数摘要、公开结果和耗时，不展示模型原始 reasoning_content。

### 4. 演示审批恢复

在 Agent 中输入：

~~~text
重建文档 doc-xxx 的索引。
~~~

Agent 会在实际写入前暂停。批准后使用相同线程恢复；重启 rag 容器后仍可通过状态接口看到待审批信息。这个流程依赖 PostgreSQL checkpoint，而不是浏览器内存。

## RAG 检索模式

### 普通 RAG

接口 /api/rag/ask 和 /api/rag/stream 是固定的两步 RAG：

~~~text
用户问题
  -> 原问题检索
  -> 拼接 context
  -> DeepSeek 生成回答
~~~

普通 RAG 默认不调用 Query Rewrite 和 HyDE。retrieval_strategy 可以选择：

- dense：Chroma 向量检索。
- hybrid：Dense + BM25，通过 RRF 融合排名。

### Agent 的 search_knowledge_base

RAG 在新 Agent 中是一个工具，而不是每轮强制执行的前置步骤：

| 模式 | 调用步骤 | 适用场景 |
| --- | --- | --- |
| standard | 原问题 -> Hybrid -> 可信度门控 | 默认、简单企业资料问题 |
| deep | Query Rewrite -> HyDE -> Hybrid -> 上下文充分性判断 | 复杂、术语不一致、跨文档问题 |

Query Rewrite 只生成适合检索的关键词；HyDE 只生成用于向量检索的假设答案，不能直接作为事实回答。最终回答仍由 Agent 根据真实检索 context 生成。

当前 Agent 的典型路径：

~~~text
模型判断是否需要工具
  -> 不需要：直接回答
  -> 需要知识库：search_knowledge_base(standard/deep)
       -> 证据足够：Agent 根据来源回答
       -> 证据不足且联网开启：search_web 兜底一次
       -> 证据不足且联网关闭：安全拒答
~~~

### Dynamic RAG 与 RAGAS 实验分支

/api/agent/dynamic-rag 是独立的旧实验入口，固定执行：

~~~text
Query Rewrite -> HyDE -> 检索 -> 上下文充分性判断 -> 回答 / 证据不足
~~~

RAGAS 评估也不经过 Agent 工具循环：

- baseline：原问题直接检索。
- hyde_rewrite：Query Rewrite + HyDE 后检索，最终回答仍使用原问题。

两套分支保留，是为了学习和对比 RAG 策略，不代表每次 Agent 问答都会自动启用 HyDE。

## Agent 工具与审批

### 只读工具

- search_knowledge_base：standard / deep 知识库检索。
- search_web：Tavily 公开网页搜索。
- list_documents、get_document_detail：文档查询。
- get_system_status：系统和向量库状态。
- list_upload_batches、get_upload_batch_detail：批次诊断。

### 写工具

- retry_upload_item：重试失败批次项。
- reindex_document：复用源文件重新解析和入库。
- delete_document：删除指定文档的向量、元数据和可选本地文件。

写工具执行顺序：

~~~text
模型选择写工具
  -> 生成 approval_id
  -> interrupt() 暂停
  -> 用户批准 / 拒绝
  -> Command(resume=...) 恢复
  -> 批准后执行一次副作用
~~~

拒绝、过期审批、重复审批和并发提交都会被服务层拦截。Agent 不可以调用任意 SQL、Shell、文件路径或清空整个知识库的工具。

## 接口速查

### Agent

| 方法 | 路径 | 用途 |
| --- | --- | --- |
| POST | /api/agent/runs/stream | 启动可恢复 Agent 运行，返回 SSE |
| POST | /api/agent/threads/{thread_id}/resume/stream | 恢复审批后的运行 |
| GET | /api/agent/threads/{thread_id}/state | 查询线程、轨迹、来源和审批状态 |
| DELETE | /api/agent/threads/{thread_id} | 删除空闲 Agent 会话 |
| POST | /api/agent/invoke | 兼容调用入口，遇到审批返回 409 |
| POST | /api/agent/dynamic-rag | Dynamic RAG 实验入口 |

启动 Agent：

~~~powershell
$body = @{
  input = "根据知识库说明报销需要哪些材料？"
  thread_id = "demo-thread"
  web_search_enabled = $false
} | ConvertTo-Json
Invoke-RestMethod -Method Post -Uri "http://127.0.0.1:8000/api/agent/runs/stream" -ContentType "application/json" -Body $body
~~~

### 文档与批次

| 方法 | 路径 | 用途 |
| --- | --- | --- |
| POST | /api/documents/upload | 单文件上传和入库 |
| POST | /api/documents/upload-batch | 批量上传、去重、状态和失败隔离 |
| GET | /api/documents | 文档列表 |
| DELETE | /api/documents/{document_id} | 删除单个文档 |
| POST | /api/documents/{document_id}/reindex | 重建单个文档索引 |
| GET | /api/documents/upload-batches?limit=20 | 批次历史 |
| GET | /api/documents/upload-batches/{batch_id} | 批次详情 |
| POST | /api/documents/upload-batches/{batch_id}/items/{item_id}/retry | 重试失败文件 |

### RAG、状态与评估

| 方法 | 路径 | 用途 |
| --- | --- | --- |
| GET | /api/health | PostgreSQL、Agent checkpoint 和模型配置状态 |
| GET | /api/status | 文档数、chunk 数、Provider 和存储信息 |
| GET | /api/vector-store/chunks | 分页查看 Chroma chunk |
| DELETE | /api/vector-store/clear?confirm=true | 清空知识库，谨慎使用 |
| POST | /api/rag/ask | 普通非流式 RAG |
| POST | /api/rag/stream | 普通流式 RAG |
| POST | /api/evaluation/ragas/build-samples | 生成 RAGAS 样本，不调用 Judge |
| POST | /api/evaluation/ragas/run | 生成样本并运行 RAGAS |

普通 RAG 请求示例：

~~~json
{
  "question": "员工报销需要准备哪些材料？",
  "thread_id": "rag-demo",
  "top_k": 3,
  "retrieval_strategy": "hybrid",
  "document_id": null,
  "filename": null
}
~~~

## RAGAS 评估

评估集 eval_data/ai_agents_in_depth_eval_dataset.json 包含围绕 AI Agent PDF 整理的问题和标准答案，共 60 条。问题集不会自动写入 Chroma；应先把对应 PDF 解析并导入知识库，再生成样本。

先用少量样本确认链路：

~~~powershell
uv run python scripts\evaluate_ragas.py --retrieval-mode baseline --max-samples 3 --build-only --force-rebuild
~~~

然后分别运行两组实验：

~~~powershell
uv run python scripts\evaluate_ragas.py --retrieval-mode baseline --retrieval-strategy dense --top-k 3 --force-rebuild
uv run python scripts\evaluate_ragas.py --retrieval-mode hyde_rewrite --retrieval-strategy dense --top-k 3 --force-rebuild
uv run python scripts\compare_evaluations.py
~~~

两种模式的调用链：

~~~text
baseline
  问题 -> 原问题检索 -> 生成回答

hyde_rewrite
  问题 -> Query Rewrite -> HyDE -> HyDE 文本检索
       -> 原问题 + context 生成回答
~~~

RAGAS 指标：

- Faithfulness：忠实度
- Answer Relevancy：回答相关性
- Context Precision：上下文精度
- Context Entity Recall：上下文实体召回
- Noise Sensitivity：噪声敏感度
- Context Recall：上下文召回
- 额外检索指标：Hit@1、Hit@3、MRR

评估结果可能受供应商限流、超时和结构化输出兼容性影响。报告必须保留空值数量；没有实际运行结果时，不要在 README 或简历中声称某种策略一定提升了准确率。完整字段、缓存规则和问题集说明见 [docs/ai-agents-in-depth-evaluation.md](docs/ai-agents-in-depth-evaluation.md) 与 [docs/ragas-ab-evaluation.md](docs/ragas-ab-evaluation.md)。

## 项目结构

~~~text
Dockerfile / compose.yaml / .dockerignore
  Docker 单机部署和构建上下文

app/
  main.py                          FastAPI 生命周期、健康检查和静态页面
  api/routes.py                    REST / SSE 接口
  schemas.py                       Pydantic 请求和响应模型
  agent/graph.py                   Agent 状态、工具循环、证据闸门和轨迹
  agent/tools.py                   知识库、网页、文档和审批工具
  agent/checkpoint.py              PostgreSQL LangGraph checkpoint 生命周期
  services/agent_service.py        Agent 运行、SSE、锁、审批和恢复
  services/document_service.py     解析、切分、Embedding 和入库
  services/upload_batch_service.py 批量状态、去重、重试和补偿
  services/rag_service.py          普通 RAG 和流式 RAG
  services/evaluation_service.py   RAGAS 样本、缓存和评估
  services/mineru_client.py        MinerU API 客户端
  rag/text_splitter.py             Markdown 和递归字符切分
  rag/embeddings.py                Qwen / OpenAI-compatible Embedding
  rag/vector_store.py              Chroma、BM25 和 RRF
  rag/llm.py                       普通 RAG Prompt 和回答
  storage/database.py              PostgreSQL 业务数据访问
  static/                          原生 HTML / CSS / JavaScript 控制台

app/workflows/
  dynamic_rag.py                   Dynamic RAG：Rewrite + HyDE + 充分性判断
  document_batch.py                LangGraph Orchestrator-Worker 批量上传
  agent_router.py                  旧版固定路由学习材料和兼容代码

scripts/
  run_local.ps1                    uv 本地启动
  smoke_test.py                    不调用真实云服务的冒烟测试
  evaluate_ragas.py               RAGAS 评估命令行入口
  compare_evaluations.py           baseline / HyDE 汇总
  evaluate_agent_tasks.py          Agent 工具行为评估
  migrate_sqlite_to_postgres.py    旧 SQLite 元数据一次性迁移

eval_data/
  ai_agents_in_depth_eval_dataset.json  60 条 PDF 评测集
  agent_tasks.json                      Agent 工具行为任务集

docs/
  agent-workbench.md               Agent 循环、联网、审批和恢复
  batch-upload-engineering.md      批量上传工程闭环
  mineru-pdf-processing.md         MinerU 大 PDF 拆分和合并
  docker-deployment.md             Docker、卷和排错
  postgresql-migration.md          SQLite 到 PostgreSQL 迁移
  ai-agents-in-depth-evaluation.md PDF 评测集说明
  project-structure.md             代码目录索引
~~~

## 测试与 CI

测试不会调用真实 MinerU、Embedding、DeepSeek 或 Tavily；外部服务使用 mock，数据库使用独立的 PostgreSQL rag_test。

~~~powershell
uv sync --frozen
uv run pytest -q
uv run python -m compileall -q app scripts tests
node --check app\static\app.js
docker compose config --quiet
uv run python scripts\smoke_test.py
~~~

也可以运行：

~~~powershell
.\scripts\test.ps1
~~~

GitHub Actions 会执行 uv sync --frozen、pytest、Python 编译检查、前端语法检查和 Docker build。真实 RAGAS 评估不放入 CI，避免消耗付费 API 和受供应商网络影响。

## 文档与学习路线

建议按下面顺序阅读和运行：

1. sample_docs/company_handbook.md、app/rag/text_splitter.py：理解文档、metadata、chunk 和标题语义。
2. app/rag/embeddings.py、app/rag/vector_store.py：理解 Embedding、Chroma、Dense、BM25 和 RRF。
3. app/services/rag_service.py、app/rag/llm.py：理解“检索 -> context -> Prompt -> 回答”的普通 RAG。
4. app/agent/graph.py、app/agent/tools.py：理解模型决策、ToolNode、有界循环、来源约束和联网开关。
5. app/services/agent_service.py、app/agent/checkpoint.py：理解 SSE、PostgreSQL checkpoint、审批中断和恢复。
6. docs/batch-upload-engineering.md、docs/mineru-pdf-processing.md：理解复杂文档入库和失败重试。
7. docs/ai-agents-in-depth-evaluation.md、docs/ragas-ab-evaluation.md：理解 RAGAS 样本、A/B 实验和指标解释。

如果只想快速体验：先用 Docker 启动，上传示例 Markdown，然后在 Agent 工作台提问；如果想学习 HyDE 和 Query Rewrite，再运行 /api/agent/dynamic-rag 或 RAGAS hyde_rewrite 实验。

## 边界与后续方向

当前项目明确不包含：

- 用户登录、RBAC、多租户和公网鉴权；
- Redis、Celery、Kubernetes 和多副本任务调度；
- PostgreSQL 高可用、读写分离和 Chroma 集群；
- 本地大模型服务；
- 没有实验数据支撑的“准确率提升”宣传。

适合继续扩展的方向：

- 增加更稳定的离线评测和按问题类型分析；
- 增加 reranker 或更严格的引用覆盖率检查；
- 将同步批量上传升级为持久化异步任务队列；
- 增加认证、权限和审计日志后再面向真实团队使用。

## License

代码使用 [MIT License](LICENSE)。示例 PDF、评测集和外部文档应遵守其原始版权和使用条款；项目不会把完整版权材料作为运行依赖提交到仓库。详见 [docs/copyright.md](docs/copyright.md)。
