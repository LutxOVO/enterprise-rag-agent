# Docker 单机部署说明

本文说明如何把当前 RAG 项目运行在 Docker 容器中。部署目标是单机学习和演示，不会把 Qwen、DeepSeek 或 MinerU 模型下载到本地。

## 1. 部署结构

```text
宿主机
  |
  +-- Docker Compose
        |
        +-- rag 容器
              |-- FastAPI + LangChain + LangGraph
              |-- SQLite
              |-- Chroma
              |-- 文档上传和 MinerU 输出
              |
              +-- 云端 Qwen Embedding
              +-- 云端 DeepSeek Chat
              +-- MinerU 官方 API
```

容器中运行的是 Python 应用。模型调用仍然通过 `.env` 中的 API Key 访问云端。宿主机的 `data/` 目录挂载到容器的 `/app/data`，因此容器删除或重建后，知识库数据不会随镜像消失。

## 2. 关键文件

- `Dockerfile`：描述如何构建应用镜像。
- `compose.yaml`：描述容器、端口、环境变量、数据卷和健康检查。
- `.dockerignore`：避免把 `.env`、`.venv`、`data/` 和缓存复制到构建上下文。
- `.env`：运行时注入 API Key，不会被复制进镜像。
- `data/`：保存 SQLite、Chroma、上传文件和 MinerU 解析结果。

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

Docker Compose 会读取 `.env` 并把这些变量注入 `rag` 容器。真实 `.env` 被 `.dockerignore` 排除，不会进入镜像层。

### 3.3 检查 Compose 配置

```powershell
docker compose config --quiet
```

命令没有输出并返回成功，表示 Compose 配置可以解析。不要把 `docker compose config` 的完整输出发到公开位置，因为它可能包含环境变量值。

### 3.4 构建并启动

```powershell
docker compose build
docker compose up -d
docker compose ps
```

第一次构建会安装 `pyproject.toml` 和 `uv.lock` 中的依赖，可能需要几分钟。修改 Python 代码后再次执行 `docker compose build`，Docker 会复用依赖缓存层。

## 4. 检查服务

Compose 健康检查访问的是 `/api/health`，不是 `/health`：

```powershell
Invoke-RestMethod http://127.0.0.1:8000/api/health
```

预期返回类似：

```json
{
  "status": "ok",
  "app": "Enterprise Knowledge Base RAG Agent"
}
```

浏览器地址：

```text
Web 控制台：http://127.0.0.1:8000/
Swagger 文档：http://127.0.0.1:8000/docs
健康检查：http://127.0.0.1:8000/api/health
```

查看实时日志：

```powershell
docker compose logs -f rag
```

## 5. 端口映射

Compose 默认把宿主机的 `8000` 映射到容器的 `8000`：

```text
宿主机 127.0.0.1:8000 -> 容器 0.0.0.0:8000
```

如果 8000 已被占用，在 `.env` 中增加或修改：

```env
RAG_PORT=8001
```

然后重新启动：

```powershell
docker compose up -d
```

此时访问 `http://127.0.0.1:8001/`。容器内部端口仍然是 8000，不需要修改 Dockerfile。

## 6. 数据持久化

Compose 中的卷映射是：

```yaml
volumes:
  - ./data:/app/data
```

应用中的 `DATA_DIR` 在容器里被设置为 `/app/data`，因此以下内容都会写到宿主机项目目录：

- `data/app.db`：SQLite 文档、批次和对话记录。
- `data/chroma/`：Chroma 向量库。
- `data/uploads/`：原始上传文件。
- `data/mineru_output/`：MinerU 输出和 PDF 分片解析结果。

停止容器：

```powershell
docker compose down
```

`docker compose down` 只删除容器和网络，不删除宿主机 `data/`。重新执行 `docker compose up -d` 后，原来的知识库仍然可以使用。

## 7. 上传和 RAG 流程

Docker 不改变现有 API。启动后可以继续使用原来的接口：

```powershell
curl.exe -X POST "http://127.0.0.1:8000/api/documents/upload" -F "file=@sample_docs/company_handbook.md"
```

文件进入容器后，仍然按原流程处理：

```text
上传文件
-> 保存到 /app/data/uploads
-> MinerU 或 pypdf 解析
-> Markdown 切分
-> Qwen Embedding
-> Chroma
-> SQLite 登记
```

超过 MinerU 单次页数限制的 PDF 仍会自动拆分。拆分后的临时文件和解析结果也位于 `/app/data`，所以重启容器不会丢失排查材料。

## 8. 常见问题

### 端口被占用

修改 `.env` 中的 `RAG_PORT`，例如设置为 `8001`，再执行 `docker compose up -d`。

### 容器启动但健康检查失败

先看状态和日志：

```powershell
docker compose ps
docker compose logs --tail=200 rag
```

常见原因是应用没有监听 `0.0.0.0`、依赖安装失败，或容器刚启动还处于 `start_period` 时间内。等待几十秒后再次执行健康检查。

### API Key 缺失

缺少 API Key 通常不会阻止 FastAPI 健康接口启动，但上传、Embedding、问答或 MinerU 解析会失败。确认 `.env` 中的变量名与 `.env.example` 完全一致，然后重建或重启容器：

```powershell
docker compose up -d --build
```

### Docker Desktop 无法挂载 data 目录

确认 Docker Desktop 已允许当前项目所在磁盘被 Linux 容器访问。Windows 下通常需要检查 Docker Desktop 的文件共享或资源访问设置。

### 修改代码后页面没有变化

Docker Compose 默认不会热重载。重新构建并启动：

```powershell
docker compose up -d --build
```

本地开发仍然可以使用原来的 `uv run uvicorn app.main:app --reload`，Docker 部署和本地开发互不冲突。

## 9. 学习建议

可以按下面顺序理解 Docker：

1. `Dockerfile` 是镜像的构建步骤。
2. 镜像是应用和依赖的只读模板。
3. 容器是镜像启动后的运行实例。
4. `ports` 负责宿主机和容器之间的网络映射。
5. `volumes` 负责把容器内数据保存到宿主机。
6. `env_file` 负责在启动时注入配置和 API Key。
7. `healthcheck` 负责告诉 Compose 应用是否已经可以接收请求。

建议先运行健康检查，再上传一个 Markdown 文件，最后停止并重新启动容器，观察 `data/` 中的 SQLite 和 Chroma 是否仍然存在。

## 10. 当前方案的边界

这是单机部署方案，适合学习、演示和个人项目：

- 不提供多副本扩展。
- 不提供 HTTPS、用户认证和反向代理。
- 不把模型下载到镜像中。
- 不使用 Redis、MySQL、Celery 或 Kubernetes。
- SQLite、Chroma 和进程内锁决定了容器默认使用单 worker。
