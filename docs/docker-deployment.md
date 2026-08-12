# Docker 单机部署说明

本文说明如何把当前 RAG 项目运行在 Docker Compose 中。部署由 FastAPI 应用和 PostgreSQL 组成，不会把 Qwen、DeepSeek 或 MinerU 模型下载到本地。

## 1. 部署结构

```text
宿主机
  |
  +-- Docker Compose
        |
        +-- rag 容器
        |     |-- FastAPI + LangChain + LangGraph
        |     |-- Chroma
        |     |-- 上传文件和 MinerU 输出
        |     +-- 云端 Qwen / DeepSeek / MinerU API
        |
        +-- postgres 容器
              +-- PostgreSQL 文档、批次、指纹和对话历史
```

PostgreSQL 使用命名卷 `postgres_data` 持久化，Chroma、原始上传文件和 MinerU 输出使用宿主机的 `data/` 目录。删除容器不会删除这两类数据。

Compose 默认只把 PostgreSQL 和 FastAPI 绑定到宿主机 `127.0.0.1`；如需对外提供服务，应在反向代理、认证和 TLS 之后再开放端口。

## 2. 关键文件

- `Dockerfile`：构建阶段使用 `uv sync --frozen` 按 `uv.lock` 安装依赖；运行阶段直接调用 `/app/.venv/bin/python -m uvicorn`。
- `compose.yaml`：定义 `rag`、`postgres`、端口、环境变量、健康检查和数据卷。
- `.dockerignore`：避免把 `.env`、`.venv`、`data/` 和缓存复制到构建上下文。
- `.env`：运行时注入 API Key 和 PostgreSQL 配置，不会被复制进镜像。
- `data/`：保存 Chroma、上传文件和 MinerU 解析结果。
- `postgres_data`：保存 PostgreSQL 业务数据。

## 3. 第一次启动

### 3.1 检查 Docker

确认 Docker Desktop 已启动，并使用 Linux containers：

```powershell
docker --version
docker compose version
```

### 3.2 准备环境变量

在项目根目录执行：

```powershell
copy .env.example .env
```

然后编辑 `.env`，至少填写：

```env
QWEN_API_KEY=你的 Qwen Embedding API Key
DEEPSEEK_API_KEY=你的 DeepSeek API Key
MINERU_API_TOKEN=你的 MinerU API Token
```

PostgreSQL 默认配置为：

```env
POSTGRES_DB=rag
POSTGRES_USER=rag
POSTGRES_PASSWORD=替换为本机密码
POSTGRES_PORT=5432
```

Compose 内部会把应用的数据库地址设置为：

```text
postgresql+psycopg://${POSTGRES_USER}:${POSTGRES_PASSWORD}@postgres:5432/${POSTGRES_DB}
```

### 3.3 检查并启动

```powershell
docker compose config --quiet
docker compose up -d --build
docker compose ps
```

预期看到 `postgres` 和 `rag` 两个服务。等待两个服务的状态变为 `healthy`。

第一次构建会安装 `pyproject.toml` 和 `uv.lock` 中的依赖，可能需要几分钟。修改 Python 代码后再次执行 `docker compose up -d --build`，Docker 会复用依赖缓存层。

## 4. 检查服务

```powershell
Invoke-RestMethod http://127.0.0.1:8000/api/health
```

预期返回：

```json
{
  "status": "ok",
  "app": "Enterprise Knowledge Base RAG Agent",
  "agent": {
    "checkpointer_ready": true,
    "model_configured": true,
    "ready": true
  }
}
```

如果 `model_configured` 为 `false`，健康检查仍可成功，但 Agent 接口会返回 `503`；上传、普通 RAG 等旧接口可以继续使用各自需要的配置。填写 `.env` 后重启 `rag` 容器。

浏览器地址：

```text
Web 控制台：http://127.0.0.1:8000/
Swagger 文档：http://127.0.0.1:8000/docs
健康检查：http://127.0.0.1:8000/api/health
```

`/api/health` 会执行 `SELECT 1`，所以它不仅检查 FastAPI 进程，也能发现 PostgreSQL 连接失败。

查看日志：

```powershell
docker compose logs -f rag
docker compose logs -f postgres
```

## 5. 端口映射

默认端口为：

```text
宿主机 127.0.0.1:8000 -> rag 容器 8000
宿主机 127.0.0.1:5432 -> postgres 容器 5432
```

如果端口被占用，在 `.env` 中修改：

```env
RAG_PORT=8001
POSTGRES_PORT=5433
```

然后重新启动：

```powershell
docker compose up -d
```

应用访问地址会变成 `http://127.0.0.1:8001/`。容器内部仍然使用 `postgres:5432`，不需要修改应用代码。

## 6. 数据持久化

Compose 使用两类存储：

```yaml
volumes:
  - postgres_data:/var/lib/postgresql/data
  - ./data:/app/data
```

PostgreSQL 命名卷保存：

- 文档元数据。
- 批次和文件状态。
- SHA-256 指纹。
- 多轮对话历史。

宿主机 `data/` 保存：

- `data/chroma/`：Chroma 向量库。
- `data/uploads/`：原始上传文件。
- `data/mineru_output/`：MinerU 输出和 PDF 分片解析结果。

停止服务：

```powershell
docker compose down
```

`docker compose down` 不会删除 `postgres_data` 或宿主机的 `data/`。不要使用 `docker compose down -v`，因为 `-v` 会删除 PostgreSQL 命名卷。

## 7. PostgreSQL 备份

命名卷解决容器重建后的持久化，但不等于备份。可以定期导出业务数据库：

```powershell
New-Item -ItemType Directory -Force backups | Out-Null
docker compose exec -T postgres pg_dump -U rag -d rag `
  > backups\rag_$(Get-Date -Format yyyyMMdd_HHmmss).sql
```

恢复前先停止应用写入，再执行：

```powershell
Get-Content backups\rag_20260812_120000.sql | docker compose exec -T postgres psql -U rag -d rag
```

## 8. 从旧 SQLite 迁移

如果项目以前使用 `data/app.db`，先停止旧应用并备份文件：

```powershell
copy data\app.db data\app.db.bak
```

先启动 PostgreSQL：

```powershell
docker compose up -d postgres
```

先进行只读检查：

```powershell
docker compose run --rm rag `
  /app/.venv/bin/python scripts/migrate_sqlite_to_postgres.py `
  --sqlite-path /app/data/app.db `
  --dry-run
```

确认行数和缺失文件列表后，执行迁移：

```powershell
docker compose run --rm rag `
  /app/.venv/bin/python scripts/migrate_sqlite_to_postgres.py `
  --sqlite-path /app/data/app.db
```

迁移内容包括 `documents`、`messages`、`upload_batches`、`upload_batch_items` 和 `document_fingerprints`。脚本使用主键和文件 hash 幂等 upsert，重复执行不会重复插入。

旧 Windows 路径会映射到容器路径，例如：

```text
D:\项目\data\uploads\a.pdf
-> /app/data/uploads/a.pdf
```

脚本不会删除旧 `data/app.db`，也不会重新生成 Chroma 向量。迁移完成后启动完整服务：

```powershell
docker compose up -d
```

如果报告某些文件不存在，需要把原始文件放回 `data/uploads/`，否则失败项无法执行重试或重建索引。

## 9. 上传和 RAG 流程

Docker 不改变现有 API：

```powershell
curl.exe -X POST `
  "http://127.0.0.1:8000/api/documents/upload" `
  -F "file=@sample_docs/company_handbook.md"
```

文件进入容器后，按以下流程处理：

```text
上传文件
-> 保存到 /app/data/uploads
-> MinerU 或 pypdf 解析
-> Markdown 切分
-> Qwen Embedding
-> Chroma
-> PostgreSQL 登记
```

Agent 的 LangGraph checkpoint 也保存在 PostgreSQL 命名卷中。`data/` 只保存 Chroma、上传文件和 MinerU 输出；因此 `docker compose restart rag` 不会丢失待审批线程。

## 10. 常见问题

### PostgreSQL 没有健康

```powershell
docker compose ps
docker compose logs --tail=200 postgres
```

确认 `.env` 中的 `POSTGRES_DB`、`POSTGRES_USER` 和 `POSTGRES_PASSWORD` 没有被修改成互相不匹配的值。首次启动需要等待几秒初始化数据库。

### 应用提示 DATABASE_URL 错误

Docker Compose 会自动给 `rag` 容器注入内部地址 `postgres:5432`。本地直接使用 Python 启动时，需要把 `.env` 中的地址改为 `localhost:5432`。

### API Key 缺失

缺少 API Key 通常不会阻止 PostgreSQL 和健康接口启动，但上传、Embedding、问答或 MinerU 解析会失败。确认变量名与 `.env.example` 完全一致，然后执行：

```powershell
docker compose up -d --build
```

### Docker Desktop 无法挂载 data 目录

确认 Docker Desktop 已允许当前项目所在磁盘被 Linux 容器访问。Windows 下通常需要检查 Docker Desktop 的文件共享或资源访问设置。

### 修改代码后页面没有变化

Docker Compose 默认不会热重载：

```powershell
docker compose up -d --build
```

## 11. 测试数据库

测试不能连接演示库 `rag`。本地测试需要一个独立的 `rag_test` 数据库：

```powershell
docker compose exec postgres createdb -U rag rag_test
$env:TEST_DATABASE_URL="postgresql+psycopg://rag:local-only-change-me@localhost:5432/rag_test"
.\scripts\test.ps1
```

如果数据库已经存在，`createdb` 的报错可以忽略。GitHub Actions 会自动启动独立 PostgreSQL service，并注入 `TEST_DATABASE_URL`。

上面的连接串中的 `local-only-change-me` 只是 `.env.example` 的本地占位密码；如果 `.env` 使用了其他密码，请替换为实际值。测试脚本会拒绝连接演示库 `rag`。

直接在宿主机运行 FastAPI 时，`.env` 中的 `DATABASE_URL` 应指向 `localhost:5432/rag`；Compose 启动应用时会覆盖为容器内部的 `postgres:5432` 地址。测试脚本必须使用 `rag_test`，不会允许误连演示库。

## 12. 当前方案边界

这是单机部署方案，适合学习、演示和个人项目：

- 不提供 PostgreSQL 高可用、读写分离和多副本扩展。
- 不提供 HTTPS、用户认证和反向代理。
- 不把模型下载到镜像中。
- 不使用 Redis、MySQL、Celery 或 Kubernetes。
- Chroma 和本地文件处理仍使容器默认使用单 worker。
