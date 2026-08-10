# 简历项目补强记录

这份文档记录本项目从“功能很多的 Demo”变成“有证据、能恢复、可比较的 RAG 项目”所采用的工程思路。

## 交付顺序

```text
真实 A/B 评估
-> 检索指标 Hit@K / MRR
-> README 架构和运行证据
-> 单文档删除 / 重建索引
-> BM25 + dense 的 RRF 混合检索
-> API 契约、结构化日志、CI
```

## 关键优化

### 1. 以实验结果证明效果

同一份 60 条问题集分别运行 `baseline` 和 `hyde_rewrite`。`retrieval_mode` 只表示查询策略；`retrieval_strategy` 单独表示 `dense` 或 `hybrid`，避免把两个变量混在一起。除了 RAGAS 六项指标，还记录 `Hit@1`、`Hit@3`、`MRR` 和每阶段耗时。

### 2. 混合检索解决关键词不稳定

纯向量检索擅长语义相似，但对错误码、编号、专有名词和精确短语可能不稳定。BM25 提供词项匹配，项目使用 Reciprocal Rank Fusion：

```text
RRF(chunk) = 1 / (60 + dense_rank) + 1 / (60 + bm25_rank)
```

不直接比较向量距离和 BM25 原始分数，而是先合并排名，避免两个分数空间不可比。中文 tokenizer 使用中文单字与二元词组，英文和数字按词保留；这是学习项目中的轻量实现。

### 3. 删除和重建索引形成生命周期闭环

- 删除：先删除 Chroma 中指定 `document_id` 的向量，再删除 SQLite 元数据、指纹和可选本地文件；路径删除前检查必须位于 `data/uploads` 或 `data/mineru_output` 下。
- 重建：复用原始文件和原 `document_id`，新 chunk 成功写入后才删除旧向量。解析失败时保留旧向量和源文件，用户可以再次重建。
- 过滤：问答请求支持按 `document_id` 或精确文件名限制检索范围。

SQLite 和 Chroma 仍然没有跨存储事务，所以删除和重建只能通过顺序操作、进程内写锁和异常补偿减少不一致窗口。

### 4. 让故障可排查

批量上传按文件保存 SHA-256、状态、失败阶段、耗时、重试次数和本地路径；请求 middleware 生成 `X-Request-ID` 并输出 JSON 日志；RAG 记录检索、Prompt 上下文整理和 LLM 生成耗时。这样一次失败可以从请求 ID 追到批次 ID、item ID 和处理阶段。

### 5. 让测试不消耗付费服务

pytest 中 MinerU、Embedding、Chroma 和 LLM 均使用 fake 或 mock。CI 执行 `uv sync --frozen`、pytest、编译检查、前端语法检查和 Docker build；真实 API 只在人工确认后运行评估命令。

## 简历表述边界

可以写“实现了批量上传、SHA-256 幂等去重、失败重试、MinerU 解析、RRF 混合检索和 RAGAS A/B 评估链路”。

只有在 `eval_outputs/ragas_ab_summary.json` 由目标 PDF 和同一批问题真实生成后，才能写具体提升百分比；在此之前不要写“准确率提升 X%”或“性能提升 X%”。
