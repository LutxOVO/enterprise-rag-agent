# 企业知识运营 Agent 与 RAG 知识库

[简体中文](README.md) | [English](README.en.md)

[![CI](https://github.com/LutxOVO/enterprise-rag-agent/actions/workflows/ci.yml/badge.svg)](https://github.com/LutxOVO/enterprise-rag-agent/actions/workflows/ci.yml)

这是一个适合 AI Agent / RAG / 大模型应用开发方向学习、面试演示和二次开发的本地 Demo 项目。默认入口是企业知识运营 Agent：模型先判断任务是否需要工具，普通交流可以直接回答，企业资料问题则调用 RAG Tool 检索并引用来源；删除、重建索引和失败重试等写操作会暂停等待人工审批，并依靠 PostgreSQL checkpoint 在服务重启后继续。系统同时保留文档入库、Hybrid 检索、Dynamic RAG、HyDE、RAGAS 评估和原生 Web 控制台。

项目不虚构真实公司经历、用户量或生产数据，定位是“可运行的 Demo 原型”和“本地知识库问答系统”。

## 架构总览

```mermaid
flowchart LR
    U[用户或 Web 控制台] --> API[FastAPI API]
    API --> AGENT[企业知识运营 Agent]
    AGENT --> MODEL{模型判断任务}
    MODEL -->|无需企业资料| DIRECT[直接回答]
    MODEL -->|需要工具| TOOLS[ToolNode 执行工具]
    TOOLS --> MODEL
    TOOLS --> APPROVAL{写操作需要审批}
    APPROVAL -->|批准后| EFFECT[重试 / 重建 / 删除]
    TOOLS -->|search_knowledge_base| RETRIEVE
    RETRIEVE --> MODEL
    RETRIEVE --> CONF{证据可信度}
    CONF -->|不足且允许联网| WEB[Tavily]
    WEB --> MODEL
    CONF -->|不足且禁止联网| REFUSE[安全拒答]
    AGENT --> CHECKPOINT[(PostgreSQL LangGraph checkpoint)]
    API --> INGEST[上传与入库流水线]
    INGEST --> PARSE[MinerU 或 pypdf 解析]
    PARSE --> SPLIT[Markdown 标题切分与递归切分]
    SPLIT --> EMB[Qwen text-embedding-v4]
    EMB --> CHROMA[(Chroma 持久化向量库)]
    RETRIEVE --> STRATEGY{检索策略}
    STRATEGY --> DENSE[Dense 向量检索]
    STRATEGY --> BM25[BM25 关键词检索]
    DENSE --> RRF[RRF 融合排序]
    BM25 --> RRF
    CHROMA --> DENSE
    CHROMA --> BM25
    RRF --> CONTEXT[上下文与真实来源 chunks]
    CONTEXT --> MODEL
    API --> POSTGRES[(PostgreSQL 状态、批次、指纹、对话)]
    API --> EVAL[RAGAS + Hit@K/MRR 离线评估]
```

## 最新 Agent 能力

- **按需 RAG 路由**：RAG 以 `search_knowledge_base` Tool 的形式暴露，模型先判断是否需要检索；寒暄、致谢和不依赖企业资料的请求不会默认消耗 Embedding 与向量检索。
- **知识证据闸门**：用户明确要求依据知识库、上传文档或企业资料回答时，如果模型漏调 RAG，LangGraph 会补一次受控检索；没有可用证据时禁止依赖模型记忆作答。
- **可恢复执行轨迹**：SSE 和 PostgreSQL checkpoint 同时记录 LLM 阶段摘要、工具调用、工具结果、来源和审批状态；页面刷新或 FastAPI 重启后仍可恢复。
- **受控联网兜底**：Tavily 由工作台开关控制，仅在知识库低可信度后执行；关闭时不会产生第三方联网请求。
- **会话生命周期管理**：支持创建、切换和删除 Agent 会话；运行中或等待审批的线程禁止删除，避免丢失未完成操作。

## 项目展示

下面的控制台截图用于快速了解项目的可演示能力。批次和检索图使用脱敏演示数据，只展示页面结构和字段，不代表真实评估结果。

![文档管理与健康状态](docs/assets/rag-console-documents.png)

![批量上传部分成功、失败阶段和重试](docs/assets/batch-upload-demo.png)

![检索来源与 hybrid RRF 调试结果](docs/assets/retrieval-sources-demo.png)

真实隔离环境演示视频（约 13 秒）：[rag-demo.webm](docs/assets/rag-demo.webm)。视频实际走通批量上传、SHA-256 重复跳过、hybrid 问答、来源 chunk、检索调试和文档删除。

RAGAS 对比图不预置虚构数字。使用目标 PDF 跑完同一批问题的 `baseline` 和 `hyde_rewrite` 后，执行 `scripts\\compare_evaluations.py` 生成真实汇总，再把脱敏后的图表放入 `docs/assets/`。

## 快速开始

下面的命令适用于 Windows PowerShell。Linux/macOS 用户可以把复制配置文件和路径命令替换成对应写法。

### 方式一：Docker Compose

Docker 会同时启动 FastAPI 和 PostgreSQL，适合第一次运行或不想单独配置 Python 环境的情况：

```powershell
git clone https://github.com/LutxOVO/enterprise-rag-agent.git
cd enterprise-rag-agent
copy .env.example .env
# 编辑 .env：至少填写 QWEN_API_KEY 和 DEEPSEEK_API_KEY；解析 PDF/Office 时再填写 MINERU_API_TOKEN
docker compose config --quiet
docker compose up -d --build
docker compose ps
```

验证服务：

```powershell
Invoke-RestMethod http://127.0.0.1:8000/api/health
```

浏览器访问 `http://127.0.0.1:8000/`，接口文档访问 `http://127.0.0.1:8000/docs`。只测试 Markdown 或纯文本 RAG 时可以暂时不填写 MinerU Token；上传复杂文档前再配置它。

### 方式二：uv 本地运行

本地运行仍然需要 PostgreSQL。可以只启动 Compose 中的数据库，再由 uv 启动 FastAPI：

```powershell
git clone https://github.com/LutxOVO/enterprise-rag-agent.git
cd enterprise-rag-agent
uv venv .venv --python 3.12
uv sync --frozen
copy .env.example .env
# 编辑 .env；本地 DATABASE_URL 使用 localhost
docker compose up -d postgres
.\scripts\run_local.ps1
```

修改 `RAG_PORT` 后，可以使用 `.\scripts\run_local.ps1 -Port 8001` 启动到其他端口。

## 配置说明

### 必填和可选密钥

| 配置 | 用途 | 什么时候需要 |
| --- | --- | --- |
| `QWEN_API_KEY` | Qwen `text-embedding-v4` 文本向量化 | 上传文档、检索和 RAG 问答 |
| `DEEPSEEK_API_KEY` | RAG 回答、Agent 路由和 RAGAS Judge | RAG 问答、Agent、RAGAS |
| `MINERU_API_TOKEN` | MinerU 云端文档解析 | PDF、Office、图片等复杂文档 |
| `TAVILY_API_KEY` | 低可信度知识库结果的联网兜底 | 打开工作台“联网搜索”开关时 |

默认使用 Qwen `text-embedding-v4` 和 DeepSeek。Embedding 只负责把文本转换为向量，DeepSeek 负责回答、Agent 工具选择和评估；不要使用聊天模型代替 Embedding 模型。

复制 `.env.example` 后只填写自己的密钥：

```powershell
copy .env.example .env
```

`.env`、数据库密码、上传文件和评测输出均不应提交到 Git。`.env.example` 中的值只是占位符。

## 技术栈

- Python、FastAPI、Pydantic、Uvicorn
- LangChain、LangGraph、Dynamic RAG、Tool
- DeepSeek Chat Model
- Qwen `text-embedding-v4` Embedding
- 阿里云百炼 OpenAI-compatible Embedding 接口
- MinerU 官方 API Token
- Chroma 向量库、PostgreSQL、SQLAlchemy、psycopg
- RAGAS、DeepSeek Judge Model
- 原生 HTML / CSS / JavaScript 前端
- uv 环境管理

## 核心功能

- 上传 `txt`、`md` 文档，复杂文档通过 MinerU API 解析。
- 支持 MinerU 可处理的 `pdf`、`doc/docx`、`ppt/pptx`、图片、HTML 等文件。
- 使用 `MarkdownHeaderTextSplitter` 保留 Markdown 标题结构，再对长文本做二次切分。
- 使用 Qwen `text-embedding-v4` Embedding 生成向量并写入 Chroma。
- 使用 Chroma retriever 做 Top-K 检索，并返回 chunk、metadata、score。
- 增加 BM25 + dense 的 RRF 混合检索，并支持按文档 ID 或文件名过滤。
- 支持普通 RAG 问答和 SSE 流式问答。
- 支持基于 `thread_id` 的简单对话历史。
- 支持显式 Dynamic RAG 工作流：自动查询重写、HyDE 虚构答案、上下文充分性判断和 system prompt 注入。
- 支持 RAGAS 评估，使用 DeepSeek 作为 Judge Model，并缓存 response / contexts 避免重复消耗 token。
- 内置 Web 控制台，可完成上传、问答、检索调试和评估操作。
- 支持单文档删除、复用原文件重建索引、请求 ID 和 RAG 阶段耗时记录。
- 默认 Agent 使用 `MessagesState` 和 `ToolNode` 进行有界多轮工具循环，最多执行 6 次工具调用。
- Agent 工作台提供“联网搜索”开关：模型先判断是否需要工具；明确要求企业资料时调用知识库，开启联网搜索后仅在知识库证据不足时使用 Tavily 兜底。
- 写工具通过 `interrupt()` 暂停，使用相同 `thread_id` 和 `Command(resume=...)` 恢复；审批状态、工具轨迹和来源可在刷新或重启后恢复。
- SSE 输出 `run_started`、`llm_stage`、`tool_call`、`tool_result`、`approval_required`、`answer`、`error`、`done` 事件；轨迹会显示模型分析阶段、工具执行和审批状态。

## 项目结构

```text
Dockerfile                     Docker 镜像构建文件
compose.yaml                   Docker Compose 单机部署配置
.dockerignore                  Docker 构建上下文排除规则
LICENSE                        项目代码许可证
CHANGELOG.md                   简历项目版本记录
app/
  api/routes.py                 FastAPI 接口
  agent/graph.py                LangGraph 工具路由
  agent/checkpoint.py           PostgreSQL AsyncPostgresSaver 生命周期
  agent/dynamic_prompt_agent.py Dynamic RAG 兼容入口
  agent/tools.py                Agent 工具
  core/config.py                环境变量和路径配置
  rag/embeddings.py             Qwen / SiliconFlow / OpenAI-compatible Embedding
  rag/llm.py                    Prompt 和模型调用
  rag/text_splitter.py          Markdown 和普通文本切分
  rag/vector_store.py           Chroma 向量库封装
  services/document_service.py  文档上传、解析、切分、入库
  services/upload_batch_service.py 批次预处理、去重、重试和响应汇总
  services/evaluation_service.py RAGAS 评估服务
  services/document_management_service.py 单文档删除和重建索引
  services/mineru_client.py     MinerU API Token 客户端
  services/rag_service.py       普通 RAG 问答流程
  services/agent_service.py     Agent 工具循环、SSE、审批和 checkpoint 恢复
  services/vector_store_service.py 向量库状态、调试和清理服务
  static/                       原生 Web 控制台
  storage/database.py           PostgreSQL 文档、对话、批次和文件指纹
eval_data/
  ai_agents_in_depth_eval_dataset.json  AI Agent PDF 专用评测集（60 条）
  ragas_ab_dataset.json         旧版公司手册 A/B 数据集（保留作示例）
  agent_tasks.json              Agent 工具行为评估集（12 条）
sample_docs/
  company_handbook.md           示例知识库文档
scripts/
  smoke_test.py                 冒烟测试
  evaluate_ragas.py             命令行 RAGAS 评估
  compare_evaluations.py        baseline / hyde_rewrite 汇总对比
  evaluate_agent_tasks.py       Agent 工具选择、审批和来源覆盖评估
  migrate_sqlite_to_postgres.py 一次性迁移旧 SQLite 元数据
tests/
  test_*.py                     上传、检索、API、重建和安全测试
docs/
  batch-upload-engineering.md   批量上传工程闭环和优化思路
  mineru-pdf-processing.md      MinerU 大 PDF 自动拆分和合并说明
  docker-deployment.md          Docker 部署、数据卷和排错说明
  postgresql-migration.md       SQLite 到 PostgreSQL 迁移说明
  ai-agents-in-depth-evaluation.md  AI Agent PDF 评测集说明和运行方法
  agent-workbench.md            Agent 循环、审批恢复、SSE 和评估说明
  resume-project-hardening.md   简历项目补强路线和技术取舍
  copyright.md                  示例数据与版权边界
```

## 环境安装

进入项目目录：

```powershell
cd enterprise-rag-agent
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
<项目目录>\.venv\Scripts\python.exe
```

## 环境变量

复制示例配置：

```powershell
copy .env.example .env
```

核心配置示例：

```env
DATABASE_URL="postgresql+psycopg://rag:local-only-change-me@localhost:5432/rag"
POSTGRES_DB="rag"
POSTGRES_USER="rag"
POSTGRES_PASSWORD="local-only-change-me"
POSTGRES_PORT=5432

EMBEDDING_PROVIDER="qwen"
CHAT_PROVIDER="deepseek"

QWEN_API_KEY="你的阿里云百炼 API Key"
# Qwen 官方兼容接口地址；域名包含 dashscope，但项目配置名统一使用 QWEN。
QWEN_BASE_URL="https://dashscope.aliyuncs.com/compatible-mode/v1"
QWEN_EMBEDDING_MODEL="text-embedding-v4"
QWEN_EMBEDDING_DIMENSIONS=1024

PDF_PARSER="mineru_api"
MINERU_API_BASE_URL="https://mineru.net/api/v4"
MINERU_API_TOKEN="你的 MinerU API Token"
MINERU_MODEL_VERSION="vlm"
MINERU_LANGUAGE="ch"
MINERU_MAX_PAGES_PER_REQUEST=200

DEEPSEEK_API_KEY="你的 DeepSeek API Key"
DEEPSEEK_BASE_URL="https://api.deepseek.com"
DEEPSEEK_CHAT_MODEL="deepseek-chat"

# 工作台开启联网搜索后，知识库证据不足时才会使用 Tavily。
TAVILY_API_KEY="你的 Tavily API Key"
TAVILY_MAX_RESULTS=5
TAVILY_SEARCH_DEPTH="basic"

CHUNK_SIZE=700
CHUNK_OVERLAP=120
TOP_K=4
EMBEDDING_BATCH_SIZE=10
KNOWLEDGE_MIN_SOURCES=2
KNOWLEDGE_MIN_CONFIDENCE=0.45
HYBRID_RRF_K=60
HYBRID_CANDIDATE_MULTIPLIER=3
```

本项目把职责拆开：DeepSeek 负责问答、Dynamic RAG、Agent 路由和 RAGAS 裁判；Qwen `text-embedding-v4` 负责文本向量化。切换 Embedding 模型后，旧向量和新向量不在同一个语义空间，必须在控制台执行“清空知识库”后重新上传文档建立索引。

### MinerU 大 PDF 的处理方式

MinerU 官方接口单次请求有页数限制。项目通过 `MINERU_MAX_PAGES_PER_REQUEST` 默认设置为 200 页：

1. 上传原始 PDF 后，先由 `pypdf` 读取页数；这里只读取 PDF 结构，不负责 OCR 或正文解析。
2. 不超过 200 页时，原 PDF 直接调用一次 MinerU。
3. 超过 200 页时，项目按最多 200 页拆成多个临时 PDF，逐片调用 MinerU。
4. 各分片的 Markdown 按页序合并，再交给后续切分、Embedding 和 Chroma 入库流程。
5. 临时分片会自动删除，原始上传文件和 MinerU 结果目录会保留，方便失败后重试。

例如 320 页 PDF 会拆成 200 页和 120 页两片，因此会产生两次 MinerU 解析请求，也会消耗两次解析额度；用户仍然只需要上传原文件一次。详细的目录、代码调用链和排错方法见：[docs/mineru-pdf-processing.md](docs/mineru-pdf-processing.md)。

## Docker 部署

Docker 部署由 FastAPI `rag` 和 PostgreSQL 两个容器组成。Qwen Embedding、DeepSeek 和 MinerU 仍然通过 `.env` 调用云端 API。PostgreSQL 使用命名卷持久化，Chroma、上传文件和 MinerU 输出挂载到宿主机的 `data/` 目录。

前置条件：安装并启动 Docker Desktop，并确认使用 Linux containers。第一次部署前复制配置文件并填写 API Key：

```powershell
cd <项目目录>
copy .env.example .env
```

然后执行：

```powershell
docker compose config --quiet
docker compose build
docker compose up -d
docker compose ps
```

健康检查接口是 `/api/health`：

```powershell
Invoke-RestMethod http://127.0.0.1:8000/api/health
```

访问地址：

```text
Web 控制台：http://127.0.0.1:8000/
Swagger 文档：http://127.0.0.1:8000/docs
健康检查：http://127.0.0.1:8000/api/health
```

查看日志和停止服务：

```powershell
docker compose logs -f rag
docker compose down
```

`docker compose down` 不会删除 `postgres_data` 命名卷或宿主机的 `data/`，因此重新启动后 PostgreSQL、Chroma 和已上传文件仍然存在。默认宿主机端口是 `8000`，数据库端口是 `5432`；如果端口被占用，可以在 `.env` 中设置 `RAG_PORT` 或 `POSTGRES_PORT` 后重新执行 `docker compose up -d`。

Dockerfile 使用依赖文件独立缓存层：修改 Python 代码时不需要重新安装全部依赖。`.env` 不会复制进镜像，而是由 Compose 在启动容器时注入。

完整的 Docker 概念、目录映射、启动流程和常见问题见：[docs/docker-deployment.md](docs/docker-deployment.md)。

### 修改代码后如何更新 Docker

`docker compose restart` 只会重启旧容器，不会把宿主机的新代码复制进镜像。修改 Python、HTML、CSS、JavaScript 或 Dockerfile 后，执行：

```powershell
docker compose up -d --build --force-recreate rag
```

只修改 `.env` 时不需要重新构建镜像，重新创建应用容器即可：

```powershell
docker compose up -d --force-recreate rag
```

验证实际运行的页面和 API：

```powershell
docker compose ps
Invoke-RestMethod http://127.0.0.1:8000/api/health
```

不要使用 `docker compose down -v`，除非你明确要删除 PostgreSQL 命名卷。普通的 `docker compose down` 不会删除 `postgres_data` 和宿主机 `data/`。浏览器仍显示旧页面时，确认访问的是 `.env` 中的 `RAG_PORT`，然后使用 `Ctrl + Shift + R` 强制刷新。

## 启动项目

推荐使用：

```powershell
./scripts/run_local.ps1 -Reload
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

- 默认打开企业知识运营 Agent 工作台：对话、工具时间线、真实来源和审批卡
- 文档上传、文档列表、系统状态查看
- RAG 实验页中的普通 RAG、流式 RAG、Dynamic RAG 问答
- 查看向量库中的 chunk
- 触发 RAGAS 样本生成和评估
- 展示 RAGAS 缓存状态和指标结果

## 常用接口

### Agent 工作台

#### 联网搜索开关

工作台输入框下方的“联网搜索”开关控制当前新一轮 Agent 任务是否允许访问公开网页：

| 状态 | Agent 行为 |
| --- | --- |
| 关闭 | 模型先判断是否需要工具。寒暄和不依赖企业资料的问题可以直接回答；明确要求依据知识库、文档或企业资料时必须调用 `search_knowledge_base`，没有证据就拒答。禁止调用 `search_web`。 |
| 开启 | 模型仍然按需选择工具；知识库检索结果低可信度时，图才自动调用一次 Tavily 兜底，不会让每个问题都联网。 |

这个开关不是只有前端显示：请求会把 `web_search_enabled` 发送到 `/api/agent/runs/stream`，并保存到该线程的 PostgreSQL checkpoint。服务端还会拦截模型越过知识库直接调用网页工具的请求，因此关闭开关时不会触发 Tavily 网络请求。对于明确要求企业资料的问题，图还会校验模型是否真正调用了知识库工具，防止模型绕过 RAG 凭记忆回答。

联网搜索结果会标记为 `source_type=web`，保留网页标题、URL、相关性分数和摘要；它们只是资料，不是可以执行的指令。Tavily 请求失败、没有结果或开关关闭时，系统不会把低可信度内容当作答案，而是安全拒答。

配置 `.env` 后需要重启本地进程或重建 `rag` 容器：

```powershell
# 本地运行
./scripts/run_local.ps1 -Reload

# Docker 运行；密钥由 .env 注入，不会写入镜像
docker compose build rag
docker compose up -d --force-recreate rag
```

`TAVILY_MAX_RESULTS` 默认是 5，`TAVILY_SEARCH_DEPTH=basic` 用于控制成本和延迟。`KNOWLEDGE_MIN_SOURCES` 与 `KNOWLEDGE_MIN_CONFIDENCE` 是演示项目的保守门槛，调整后应重新观察检索结果和 Agent 轨迹，不要把它们当成通用准确率阈值。Hybrid 的 RRF 分数只负责融合排序；可信度门控使用 Chroma 返回的原始 cosine distance，避免“有排名就误认为相关”。

### LLM 思考阶段轨迹

工作台的“轨迹”会按顺序展示“模型分析 -> 工具调用 -> 工具结果 -> 模型分析 -> 最终回答”。`llm_stage` 只返回安全的阶段摘要、耗时和 `reasoning_available` 标志，不返回 DeepSeek 原始 `reasoning_content`，后者只保存在消息内部用于下一轮工具调用兼容。默认配置 `DEEPSEEK_THINKING_TYPE=enabled`，开启会增加耗时和 Token 消耗；改为 `disabled` 后仍会显示模型分析阶段，但思考信号显示为不可用。修改 `.env` 后需要重启本地进程或重新创建 Docker 应用容器。

启动可恢复 Agent 运行：

```http
POST /api/agent/runs/stream
Content-Type: application/json

{"input":"列出当前文档，并分析最近失败批次","thread_id":"demo-thread","web_search_enabled":false}
```

恢复审批：

```http
POST /api/agent/threads/{thread_id}/resume/stream
Content-Type: application/json

{"approval_id":"审批卡中的 ID","decision":"approve","reason":null}
```

刷新页面或服务重启后查询状态：

```http
GET /api/agent/threads/{thread_id}/state
```

删除 Agent 会话：

```http
DELETE /api/agent/threads/{thread_id}
```

删除只清理该 Agent 线程的 PostgreSQL checkpoint 和浏览器本地会话数据，不删除旧 `/api/rag/*` 使用的 `messages` 表。运行中或等待审批的线程返回 `409`；不存在的线程按幂等成功处理。工作台的“最近会话”列表和垃圾桶按钮使用这个接口。

SSE 的 `data` 是 JSON，事件顺序通常是：

```text
run_started -> llm_stage -> tool_call -> tool_result -> llm_stage -> answer -> done
```

写操作会在 `tool_call` 后收到 `approval_required`，此时 `done.status` 是 `awaiting_approval`。批准后使用同一个 `thread_id` 和 `approval_id` 调用 resume；重复审批返回 `409`。Agent 的详细设计、状态字段和 Docker 重启验收见：[docs/agent-workbench.md](docs/agent-workbench.md)。

兼容接口 `/api/agent/invoke` 仍然保留；它内部调用新图，遇到审批时返回 `409`，普通 RAG 和 Dynamic RAG 仍分别由 `/api/rag/*`、`/api/agent/dynamic-rag` 提供。

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

删除或重建单个文档：

```powershell
curl.exe -X DELETE "http://127.0.0.1:8000/api/documents/<document_id>?delete_files=true"
curl.exe -X POST "http://127.0.0.1:8000/api/documents/<document_id>/reindex"
```

重建索引会复用 `data/uploads/` 中的原文件，成功写入新 chunk 后再删除旧 chunk；失败时保留源文件并保留旧索引。

### 批量上传与工程闭环

批量接口会先分块保存文件并计算 SHA-256，再按“解析 -> 切分 -> Embedding -> Chroma -> PostgreSQL 登记”的阶段处理。相同内容会标记为 `duplicate` 并计入 `skipped`，不会重复调用外部解析和向量服务。

```http
POST /api/documents/upload-batch
```

批次响应保留 `total`、`succeeded`、`failed`、`results`、`errors` 和 `graph_path`，并增加：

- `batch_id`：批次唯一 ID。
- `status`：`running`、`completed`、`partial_success` 或 `failed`。
- `skipped`：重复文件数量。
- `duration_ms`：批次耗时。
- `items`：每个文件的状态、chunk 数、耗时、失败阶段和错误信息。

查询最近批次和明细：

```http
GET /api/documents/upload-batches?limit=20
GET /api/documents/upload-batches/{batch_id}
```

只重试失败文件：

```http
POST /api/documents/upload-batches/{batch_id}/items/{item_id}/retry
```

默认上传限制和重试次数可以在 `.env` 中调整：

```env
MAX_BATCH_FILES=10
MAX_UPLOAD_FILE_SIZE_MB=50
UPLOAD_READ_CHUNK_SIZE=1048576
UPLOAD_CONCURRENCY=2
MAX_UPLOAD_RETRIES=3
```

完整的设计理由、状态流转和面试表达见：[docs/batch-upload-engineering.md](docs/batch-upload-engineering.md)。

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
  "top_k": 3,
  "retrieval_strategy": "hybrid",
  "document_id": null,
  "filename": null
}
```

`retrieval_strategy=dense` 是纯向量检索，`retrieval_strategy=hybrid` 会把 dense 和 BM25 的候选用 RRF 合并。`document_id` 和 `filename` 是可选的精确过滤条件。响应中的 `timings_ms` 会记录 `retrieval_ms`、`context_ms`、`llm_ms` 和 `total_ms`。

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
-> DeepSeek 查询重写
-> DeepSeek 生成 HyDE 虚构答案
-> Chroma retriever Top-K 检索
-> retrieved chunks 注入 system prompt
-> DeepSeek LLM 生成回答
```

### RAGAS 评估

RAGAS 支持两种检索模式，使用同一份问题集进行 A/B 对比：

- `baseline`：用户问题直接检索。
- `hyde_rewrite`：先 Query Rewrite，再生成 HyDE 假答案，最后使用 HyDE 文本检索；最终回答仍使用原始问题和统一 Prompt。

命令行分别运行两组：

默认评测集是围绕 `AI-Agents-in-Depth-zh-CN.pdf` 整理的 60 条问题。开始评测前，必须先把这份 PDF 通过 MinerU 解析并写入当前 Chroma；评测集本身不会自动进入向量库。完整流程和字段说明见 [docs/ai-agents-in-depth-evaluation.md](docs/ai-agents-in-depth-evaluation.md)。

先用 3 条样本检查检索内容：

```powershell
uv run python scripts\evaluate_ragas.py `
  --retrieval-mode baseline `
  --max-samples 3 `
  --build-only `
  --force-rebuild
```

确认样本来自目标 PDF 后，再运行两组完整评估：

```powershell
uv run python scripts\evaluate_ragas.py --retrieval-mode baseline --top-k 3 --force-rebuild
uv run python scripts\evaluate_ragas.py --retrieval-mode hyde_rewrite --top-k 3 --force-rebuild

# 需要比较 hybrid 检索时使用独立输出文件，不覆盖 dense 结果。
uv run python scripts\evaluate_ragas.py --retrieval-mode baseline --retrieval-strategy hybrid --top-k 3 --force-rebuild
```

两组默认保存到不同文件，不会互相覆盖：

```text
eval_outputs/ragas_samples_baseline.json
eval_outputs/ragas_result_baseline.json
eval_outputs/ragas_samples_hyde_rewrite.json
eval_outputs/ragas_result_hyde_rewrite.json
```

只生成样本，不调用 RAGAS 指标裁判（但仍会调用 DeepSeek 生成回答）：

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
  "dataset_path": "eval_data/ai_agents_in_depth_eval_dataset.json",
  "retrieval_mode": "baseline",
  "retrieval_strategy": "dense",
  "samples_output_path": "eval_outputs/ragas_samples_baseline.json",
  "result_output_path": "eval_outputs/ragas_result_baseline.json",
  "top_k": 3,
  "max_samples": 1,
  "force_rebuild": false
}
```

字段说明：

- `max_samples=1`：只评估前 1 条，适合先测试。
- `force_rebuild=false`：复用缓存的 response 和 retrieved_contexts，避免重复消耗 token。
- `force_rebuild=true`：重新检索并重新生成回答。
- `retrieval_mode=baseline`：直接用原问题检索。
- `retrieval_mode=hyde_rewrite`：Query Rewrite + HyDE 后再检索。
- `retrieval_strategy=dense`：只使用 Chroma dense 检索。
- `retrieval_strategy=hybrid`：使用 BM25 + dense，并通过 RRF 合并排名。

评估指标：

- 忠实度 (Faithfulness)
- 回答相关性 (Answer Relevancy)
- 上下文精度 (Context Precision)
- 上下文实体召回 (Context Entity Recall)
- 噪声敏感度 (Noise Sensitivity)
- 上下文召回 (Context Recall)
- 检索命中率 `Hit@1`、`Hit@3` 和 `MRR`

### 10 条本地 A/B 实验结果示例

以下结果是一次本地实验：同一份 AI Agent PDF、同一批 10 条问题、`top_k=3`、`dense` 检索和 DeepSeek 生成模型。它只用于展示报告格式和分析方式，不是项目的固定准确率。`eval_outputs/` 被 `.gitignore` 排除，公开仓库不会携带本地评测缓存；运行命令后可以重新生成同名文件。

| 指标 | baseline | hyde_rewrite | delta |
|---|---:|---:|---:|
| Faithfulness | 0.9429（5/10） | 1.0000（1/10） | +0.0571* |
| Answer Relevancy | 0.8878（10/10） | 0.6791（10/10） | -0.2087 |
| Context Precision | 0.8917（10/10） | 0.9833（10/10） | +0.0916 |
| Context Entity Recall | -（0/10） | -（0/10） | - |
| Noise Sensitivity | -（0/10） | -（0/10） | - |
| Context Recall | 0.6583（10/10） | 0.6917（10/10） | +0.0334 |
| Hit@1 | 0.1000 | 0.0000 | -0.1000 |
| Hit@3 | 0.3000 | 0.2000 | -0.1000 |
| MRR | 0.1833 | 0.0833 | -0.1000 |

初步结论：这批问题中，HyDE + Query Rewrite 提高了上下文精度和上下文召回，但降低了回答相关性、Hit@K 和 MRR，不能写成“整体效果提升”。Faithfulness 的有效样本数不足，Context Entity Recall 和 Noise Sensitivity 因 DeepSeek Judge 请求超时全部缺失，后续需要继续优化评估稳定性再扩展到 60 条。

\* Faithfulness 的分数只对成功返回的样本求平均，括号内是有效样本数 / 总样本数。

RAGAS 0.4.x 的 `AnswerRelevancy` 默认会请求多组生成，而 DeepSeek 兼容接口只支持 `n=1`；项目已将 `strictness` 固定为 1，并限制 Judge 并发、超时和重试。云端评估仍可能因为供应商限流或结构化输出超时产生空值，报告中必须保留空值数量。

输出文件：

```text
eval_outputs/ragas_samples_baseline.json
eval_outputs/ragas_result_baseline.json
eval_outputs/ragas_result_baseline.csv
eval_outputs/ragas_samples_hyde_rewrite.json
eval_outputs/ragas_result_hyde_rewrite.json
eval_outputs/ragas_result_hyde_rewrite.csv
eval_outputs/ragas_result_baseline.summary.json
```

生成 baseline 和 hyde_rewrite 两组后，用本地脚本汇总 RAGAS、Hit@K、MRR、空值数、耗时和问题类型：

```powershell
uv run python scripts\compare_evaluations.py
```

脚本只读取已经生成的 JSON，不调用 MinerU、Embedding 或 DeepSeek；缺少 hyde 文件时会直接报错，不能用一组结果冒充 A/B 结论。

## 测试

当前测试不调用真实 MinerU、Embedding 或 DeepSeek，外部服务全部使用 mock；真实 A/B 评估命令单独运行。

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

完整单元/API 契约测试：

```powershell
.\scripts\test.ps1
```

脚本默认使用独立的 `rag_test` 数据库，并同时执行 Python 编译检查和前端 JavaScript 语法检查。运行前请确认 PostgreSQL 已启动；如果数据库地址不同，可以先设置 `TEST_DATABASE_URL`。

GitHub Actions 会重复执行依赖锁定、pytest、编译检查、前端语法检查和 Docker build。

## 招聘展示材料

仓库已经放入三张控制台展示图；真实实验材料应在实际运行后再补充到 `docs/assets/`：

- 批量上传状态表：包含 `partial_success`、失败阶段和重试按钮；
- 检索结果：包含来源文件、chunk、dense/hybrid 策略和耗时；
- RAGAS A/B 汇总：只提交 `ragas_ab_summary.json/csv` 和少量示例，不提交整份 PDF、API Key 或原始回答全集；
- 60 秒以内的上传 -> 检索 -> 问答演示 GIF 或视频。

仓库中的评估脚本不会伪造截图或提升百分比；只有在你用目标 PDF 实际跑完两组实验后，才把真实数字写入简历。

## 简历亮点

- 构建完整 RAG 流程：文档解析、chunk 切分、Embedding、向量检索、Prompt 增强生成。
- 使用 MinerU 官方 API Token 处理 PDF、Office、图片等复杂文档。
- 使用 Chroma retriever 实现 Top-K 语义检索，并返回 chunk score 便于调试。
- 实现 Dynamic RAG + Query Rewrite + HyDE，提高检索语义匹配能力并返回可观测 graph_path。
- 使用 RAGAS + DeepSeek 对 RAG 回答和检索效果进行离线评估，并加入样本缓存降低评估成本。
- 使用 SQLAlchemy + psycopg 连接池持久化文档元数据、会话历史、批次状态和 SHA-256 文件指纹；通过事务、幂等 upsert、状态约束和 SQLite 到 PostgreSQL 一次性迁移脚本增强数据一致性。
- 提供 FastAPI 接口和原生 Web 控制台，方便演示完整工作流。
