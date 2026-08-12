# SQLite 到 PostgreSQL 迁移说明

项目当前运行时只使用 PostgreSQL。旧版 `data/app.db` 只作为一次性迁移来源，不会在新版本启动时自动读取。

运行时 PostgreSQL 保存文档元数据、会话历史、上传批次、文件级状态和 SHA-256 指纹；Chroma 向量和原始文件仍位于 `data/`。因此 PostgreSQL 是业务状态存储，不是 LangGraph PostgreSQL Checkpointer；当前 LangGraph 图仍通过 `graph.compile()` 编译。

## 当前 schema

应用启动时的 `init_db()` 会创建表并执行 `schema_migrations` 中记录的幂等迁移。当前 schema 版本为 `3`，包括：

- 时间字段使用 `TIMESTAMPTZ`，文档额外维护 `updated_at`。
- 批次 `graph_path` 使用 `JSONB`。
- 批次、item 和文件指纹增加状态 `CHECK` 约束。
- 批次 item 保存显式 `item_order`，详情接口按原始上传顺序返回文件。
- 增加消息历史、文档创建时间、批次状态和文件指纹的查询索引。

旧版 PostgreSQL 表中的 `TEXT` 时间和 JSON 文本会在启动时原地转换；迁移失败会回滚当前事务。升级后可用下面的命令确认版本和字段类型：

```powershell
docker compose exec -T postgres psql -U rag -d rag -c "SELECT MAX(version) FROM schema_migrations;"
docker compose exec -T postgres psql -U rag -d rag -c "\d+ documents"
```

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

迁移脚本先调用 `init_db()`，确保目标表和 schema 版本就绪，再使用主键和 `file_hash` 做幂等 upsert；重复运行不会创建重复记录。旧路径中的 `data/` 部分会映射到容器的 `/app/data/`：

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

迁移脚本不会导入 LangGraph checkpoint，也不会重新生成 Chroma 向量；如果原始文件或向量目录缺失，只能先补回对应的 `data/` 文件再执行重试或重建索引。

## 从空库开始

如果不需要旧数据，可以不执行迁移脚本，直接运行：

```powershell
docker compose up -d --build
```

PostgreSQL 会自动建表，之后重新上传文档即可。`docker compose down` 不会删除数据库；只有明确执行 `docker compose down -v` 才会删除 `postgres_data`。

测试必须使用独立的 `rag_test` 数据库，不能直接清理演示库 `rag`：

```powershell
docker compose exec -T postgres createdb -U rag rag_test
$env:TEST_DATABASE_URL="postgresql+psycopg://rag:local-only-change-me@localhost:5432/rag_test"
.\scripts\test.ps1
```

如果 `.env` 中使用了其他密码，请同步替换连接串中的密码。
