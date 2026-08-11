# SQLite 到 PostgreSQL 迁移说明

项目当前运行时只使用 PostgreSQL。旧版 `data/app.db` 只作为一次性迁移来源，不会在新版本启动时自动读取。

## 迁移前准备

先停止旧版应用，避免迁移过程中 SQLite 继续写入，然后备份数据库：

```powershell
copy data\app.db data\app.db.bak
```

确认旧的 Chroma、上传文件和 MinerU 输出仍然位于当前项目的 `data/` 目录。迁移脚本不会重新生成向量，原来的 `document_id` 必须保持不变。

## 只读检查

启动 PostgreSQL：

```powershell
docker compose up -d postgres
```

执行 dry-run：

```powershell
docker compose run --rm rag `
  /app/.venv/bin/python scripts/migrate_sqlite_to_postgres.py `
  --sqlite-path /app/data/app.db `
  --dry-run
```

dry-run 只读取 SQLite，输出每张表的行数和迁移后找不到的文件路径，不会写入 PostgreSQL。

## 执行迁移

确认检查结果后执行：

```powershell
docker compose run --rm rag `
  /app/.venv/bin/python scripts/migrate_sqlite_to_postgres.py `
  --sqlite-path /app/data/app.db
```

脚本会迁移：

- `documents`：文档元数据。
- `messages`：多轮对话历史。
- `upload_batches`：批次汇总状态。
- `upload_batch_items`：每个文件的处理状态、错误阶段和重试次数。
- `document_fingerprints`：SHA-256 内容指纹。

迁移使用主键和 `file_hash` 做幂等 upsert，重复运行不会创建重复记录。旧路径中的 `data/` 部分会映射到容器的 `/app/data/`：

```text
D:\项目\data\uploads\a.pdf
-> /app/data/uploads/a.pdf
```

如果输出缺失文件，需要把原文件放回 `data/uploads/`，否则该文档无法重试或重建索引。脚本不会删除 `data/app.db`。

## 迁移后验证

启动完整服务：

```powershell
docker compose up -d
docker compose ps
Invoke-RestMethod http://127.0.0.1:8000/api/health
```

随后检查：

1. `/api/documents` 中的文档数量。
2. `/api/documents/upload-batches?limit=20` 中的批次历史。
3. `/api/status` 中的 `database_path` 是否为 `postgresql`。
4. RAG 检索是否仍能返回原来的 Chroma 来源 chunk。

## 从空库开始

如果不需要旧数据，可以不执行迁移脚本，直接运行：

```powershell
docker compose up -d --build
```

PostgreSQL 会自动建表，之后重新上传文档即可。`docker compose down` 不会删除数据库；只有明确执行 `docker compose down -v` 才会删除 `postgres_data`。
