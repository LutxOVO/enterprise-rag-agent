# 企业知识运营 Agent

本项目现在有两条入口：默认入口是“企业知识运营 Agent”，普通 RAG、Dynamic RAG、文档管理、检索调试和 RAGAS 仍作为辅助实验页面。Agent 不是把一次路由包装成聊天，而是使用 LangGraph 保存状态并执行一个有界工具循环：

```text
用户任务
  -> 模型分析任务
  -> 模型选择工具
  -> ToolNode 执行
  -> 记录工具轨迹和来源
  -> 模型分析工具结果后继续选择
  -> 最终回答 / 审批中断 / 安全结束
```

## 为什么要用 checkpoint

Agent 的 `messages`、`run_id`、LLM 阶段轨迹、工具轨迹、来源、调用次数和审批状态都保存在 PostgreSQL 的 LangGraph checkpoint 中。模型对象、连接、锁和 `Path` 不进入图状态，因此状态可以被序列化，也可以在 FastAPI 容器重启后恢复。

## LLM 阶段轨迹

每次模型调用前，`begin_llm_stage` 节点会通过 LangGraph `get_stream_writer()` 发出 `llm_stage(status=started)`；模型返回后更新同一个 `stage_id`，记录完成状态、阶段摘要、工具名、耗时和 `reasoning_available`。前端因此可以看到：

```text
模型分析 -> 工具调用 -> 工具结果 -> 模型分析 -> 最终回答
```

`reasoning_available` 只表示模型响应中是否包含 DeepSeek `reasoning_content`。原始内容不会进入 SSE 或状态接口，只保留在 `AIMessage.additional_kwargs` 中，用于下一轮 DeepSeek 工具调用时回传兼容。关闭 DeepSeek thinking 模式后，生命周期阶段仍然存在，但该标志为 `false`。

模型阶段和工具阶段共享 `trace_order`，并通过独立的 `llm_trace`、`tool_trace` 字段保存。刷新页面或服务重启后，服务端返回两组轨迹，前端按序合并恢复。

应用启动时由 `app/agent/checkpoint.py` 创建一个 `AsyncPostgresSaver`，执行一次 `setup()`，然后把它传给 `build_agent_graph()`。图在启动阶段编译一次；每次请求只通过 `thread_id` 找回对应 checkpoint。

旧的 PostgreSQL `messages` 表仍然服务 `/api/rag/*` 和旧兼容接口。新的 Agent 对话历史以 LangGraph checkpoint 为准，避免同一轮对话被重复写入两套历史。

## 工具边界

Agent 可以使用以下工具：

- `search_knowledge_base`：`standard` 使用 Hybrid，`deep` 使用 Query Rewrite + HyDE + 上下文充分性判断。
- `search_web`：使用 Tavily 查询公开网页；只允许作为知识库低可信度时的兜底工具。
- `list_documents`、`get_document_detail`、`get_system_status`。
- `list_upload_batches`、`get_upload_batch_detail`。
- `retry_upload_item`、`reindex_document`、`delete_document`。

最后三项会修改知识库。Agent 不能看到清空知识库、任意 SQL、Shell 或任意文件路径工具。

工具调用上限由 `AGENT_MAX_TOOL_CALLS` 控制，默认 6 次。模型绑定工具时设置 `parallel_tool_calls=False`，让每次调用都能观察前一个工具结果。超过上限后图会安全结束，不再执行新的工具。

## 联网搜索开关

工作台的“联网搜索”开关对应 Agent 状态中的 `web_search_enabled`，只影响当前新发起的任务；前端会把它放在启动请求体中：

```json
{
  "input": "解释知识库之外的一个概念",
  "thread_id": "demo-thread",
  "web_search_enabled": false
}
```

### 关闭时：按需知识库模式

关闭后，`build_agent_system_prompt(False)` 会把以下边界放入本轮 system message：

- 不能调用 `search_web`；
- 寒暄、致谢和不依赖企业资料的问题可以直接回答；
- 如果用户明确要求依据知识库、上传文档、企业制度或项目资料，必须调用 `search_knowledge_base`；
- 调用知识库后只能依据它返回的 `context` 和来源回答，证据不足时必须明确说知识库信息不足。

提示词不是唯一防线。图在模型节点之后还会检查工具调用：即使模型错误地产生了 `search_web` call，`guard_web_search` 也只返回一个“联网搜索已关闭”的工具结果，不会创建 Tavily 客户端，更不会发出网络请求。对于明确要求企业资料的问题，如果模型没有主动生成 `search_knowledge_base` 调用，图会补一次受控的标准检索；检索结果没有可用证据时，`finalize` 会把模型可能生成的答案覆盖成安全拒答。因此 RAG 是按需工具，但不会被知识库问题绕过。

### 开启时：低可信度兜底

开启后并不是每个问题都联网，也不是每个问题都先检索知识库。流程是：

```text
模型判断是否需要工具
  -> 不需要：直接回答
  -> 需要知识库：search_knowledge_base
  -> 计算来源数量和相关性信号
  -> fallback_to_web_search=true ?
       -> 是：自动调用一次 search_web
       -> 否：继续让模型依据知识库回答
```

`standard` 默认使用 Hybrid 检索。当前演示判定为低可信度的情况包括：没有来源、来源数量低于 `KNOWLEDGE_MIN_SOURCES`，或 Chroma 的最佳 cosine distance 转换后的置信度低于 `KNOWLEDGE_MIN_CONFIDENCE`。Hybrid 的 RRF `score` 只表示融合排名贡献，不能单独证明检索内容相关。`deep` 则使用现有上下文充分性判断。这个信号是项目内的门控，不等价于事实正确率。

网页结果包含 `source_type=web`、`title`、`url`、`score` 和 `content_preview`，会进入来源列表和工具轨迹。网页内容会被明确标记为资料，不能执行网页中要求修改系统、泄露密钥或调用其他工具的文字。

Tavily 配置写在 `.env`，不要提交到 Git：

```env
TAVILY_API_KEY="你的 Tavily API Key"
TAVILY_MAX_RESULTS=5
TAVILY_SEARCH_DEPTH="basic"
```

关闭开关不消耗 Tavily 额度；开启后只有低可信度知识库检索才会产生联网调用。Tavily 请求超时、额度不足或返回空结果时，工具返回结构化失败信息，最终仍然拒答，不会把失败结果当成可靠证据。

## 审批和恢复

写工具的顺序是：

```text
模型选择写工具
  -> 工具生成 approval_id 和审批 JSON
  -> interrupt(payload)，暂停
  -> 前端显示审批卡
  -> 用户 approve/reject
  -> 使用相同 thread_id + Command(resume=...)
  -> 批准后执行一次副作用
```

`interrupt()` 前不能做不可逆副作用，因为 LangGraph 恢复时会从中断节点重新执行。项目中的写工具只在批准结果返回后调用上传重试、重建索引或删除服务。

审批卡包含工具名、参数、目标、风险等级、审批 ID 和过期时间。服务层会检查：

- 当前线程确实存在待审批任务；
- `approval_id` 与 checkpoint 中的审批 ID 一致；
- 审批没有过期；
- 同一线程没有同时提交两个运行或两个恢复请求。

已经处理的审批再次提交返回 `409`。过期审批返回 `409`，不会执行写操作。

## SSE 接口

启动新任务：

```http
POST /api/agent/runs/stream
Content-Type: application/json

{"input":"列出当前文档并说明最近失败批次","thread_id":"demo-thread","web_search_enabled":false}
```

恢复审批：

```http
POST /api/agent/threads/demo-thread/resume/stream
Content-Type: application/json

{"approval_id":"...","decision":"approve","reason":null}
```

状态查询：

```http
GET /api/agent/threads/demo-thread/state
```

SSE `data` 中的 `event` 只使用以下值：

| 事件 | 用途 |
| --- | --- |
| `run_started` | 记录本轮 run_id 和是否恢复 |
| `llm_stage` | 模型分析阶段的开始/完成/失败摘要、耗时和思考信号 |
| `tool_call` | 工具名称和参数 |
| `tool_result` | 执行摘要、状态和脱敏结果 |
| `approval_required` | 审批卡数据 |
| `answer` | 最终回答和来源 |
| `error` | 结构化错误 |
| `done` | 本轮结束，状态为 `completed`、`awaiting_approval` 或 `failed` |

`/api/agent/invoke` 保留兼容，内部同样走新图；如果遇到审批，会直接返回 `409` 和待审批信息。

## Docker 重启验收

在 `.env` 配好 PostgreSQL 和模型 key 后：

```powershell
docker compose up -d --build
Invoke-RestMethod http://127.0.0.1:8000/api/health
```

用 Agent 工作台提交“重试失败文件”任务，收到审批卡后重启 FastAPI 容器：

```powershell
docker compose restart rag
docker compose ps
```

刷新页面或直接调用状态接口，确认同一个 `thread_id` 仍返回 `awaiting_approval`。然后使用原来的 `approval_id` 调用 resume 接口。批准后应该只执行一次，状态变成 `completed`；重复调用 resume 应返回 `409`。

这次验收只调用项目自己的 PostgreSQL、批次和文件操作。测试审批逻辑时可以使用本地失败批次，避免为了演示而调用真实 MinerU 或 Embedding。

## Agent 行为评估

`eval_data/agent_tasks.json` 提供 12 条任务，覆盖知识问答、跨文档比较、系统查询、批次诊断、证据不足和三类审批写操作。每条任务约束：

- `expected_tools`：允许的主要工具；
- `forbidden_tools`：绝不能触发的危险工具；
- `requires_approval`：是否必须出现审批；
- `expected_sources`：是否需要真实来源；
- `expected_status`：预期结束状态。

生成填写模板：

```powershell
uv run python scripts\evaluate_agent_tasks.py --template
```

在 `eval_outputs/agent_results.json` 填入实际运行记录后汇总：

```powershell
uv run python scripts\evaluate_agent_tasks.py
```

脚本输出工具选择准确率、危险工具误触发率、审批触发准确率、来源覆盖率、任务完成率、平均工具调用次数和错误率。它评估 Agent 的工具行为，不替代 RAGAS 对回答忠实度和相关性的评估。
