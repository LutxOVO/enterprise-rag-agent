# LangGraph Agent 升级记录

这份文档用于记录本项目从“关键词规则路由 Agent”升级到“LLM 决策路由 + 显式条件边 LangGraph 工作流”的过程，方便后续复盘、继续迭代和面试讲解。

## 升级背景

原来的 Agent 图比较简单：

```text
route_intent -> run_tool -> END
```

其中：

- `route_intent` 通过关键词判断用户意图，并把结果写入 `state["route"]`。
- `run_tool` 通过 `if/elif/else` 判断 `route`，再执行对应工具。

原来的工具选择逻辑是：

```text
状态 / status / 系统
  -> query_system_status

文档列表 / documents / 文件列表 / 有哪些文档
  -> query_document_list

其他问题
  -> search_knowledge_base
```

这种写法适合初学阶段理解 Agent 流程，但它有两个明显问题：

- 决策逻辑是关键词规则，不具备真正的语义理解能力。
- 图结构本身是线性的，业务分支隐藏在 `run_tool` 函数内部，LangGraph 的条件边能力没有体现出来。

## 本次升级内容

本次升级完成了两件事：

1. 将 `route_intent` 从关键词判断升级为 LLM 结构化路由。
2. 将 `run_tool` 拆成多个独立节点，并使用 `add_conditional_edges` 显式表达分支。

升级后的工作流变成：

```text
route_intent
  -> query_status
  -> END

route_intent
  -> query_documents
  -> END

route_intent
  -> search_knowledge
  -> END
```

也就是说，现在 LangGraph 图结构本身就能看出三条分支，而不是所有逻辑都塞在一个工具执行函数里。

## LLM 结构化路由

现在 `route_intent` 会调用配置好的聊天模型，让 LLM 返回一个固定结构：

```python
class RouteDecision(BaseModel):
    route: Literal["status", "documents", "knowledge"]
    reason: str
```

其中：

- `route` 表示下一步要走哪条分支。
- `reason` 表示 LLM 为什么这么判断。

可选路由含义如下：

```text
status
  查询系统状态，例如文档数量、chunk 数量、向量库路径、应用状态。

documents
  查询已上传文档列表，例如知识库里有哪些文件。

knowledge
  查询知识库内容，例如公司制度、文档内容、业务知识问答。
```

这样，Agent 不再只能靠关键词判断。例如用户问：

```text
帮我看看现在知识库规模怎么样
```

关键词规则可能不一定稳定命中，但 LLM 可以理解这是在问系统状态，从而选择：

```text
route = status
```

## 条件边改造

原来 `run_tool` 内部是这样的思路：

```python
if route == "status":
    ...
elif route == "documents":
    ...
else:
    ...
```

升级后，工具执行被拆成三个节点：

```text
query_status
query_documents
search_knowledge
```

然后通过 LangGraph 的条件边控制走向：

```python
graph.add_conditional_edges(
    "route_intent",
    route_to_tool_node,
    {
        "status": "query_status",
        "documents": "query_documents",
        "knowledge": "search_knowledge",
    },
)
```

这表示：

```text
route_intent 执行完成后
  -> 读取 state["route"]
  -> 如果是 status，进入 query_status
  -> 如果是 documents，进入 query_documents
  -> 如果是 knowledge，进入 search_knowledge
```

这种写法更符合 LangGraph 的核心思想：让图结构表达工作流，而不是只在普通 Python 函数里隐藏流程分支。

## State 的变化

原来的 `AgentState` 包含：

```python
input
thread_id
route
output
tool_used
```

升级后增加了：

```python
route_reason
graph_path
```

现在 state 中会记录：

```text
input
  用户输入

thread_id
  会话 ID

route
  LLM 或 fallback 路由器选择的分支

route_reason
  选择该分支的原因

graph_path
  本次执行经过的完整图节点路径

output
  工具执行结果

tool_used
  实际调用的工具名称
```

这样每次调用 `/api/agent/invoke` 时，不仅能看到用了哪个工具，还能看到为什么走这条路线。

## FastAPI 返回变化

接口 `/api/agent/invoke` 的返回结果中，`extra` 现在包含：

```json
{
  "route": "status",
  "route_reason": "The user asked about current system status.",
  "graph_path": ["route_intent", "query_status", "END"]
}
```

这对调试很有用，因为可以直接从接口响应中看到 Agent 的决策过程和完整图路径。

## Fallback 机制

为了保证本地 demo 稳定，本次升级保留了关键词 fallback。

如果出现以下情况：

- 没有配置 Qwen / OpenAI API Key。
- LLM 路由调用失败。
- 模型结构化输出异常。

系统会自动退回到原来的关键词路由逻辑：

```text
状态 / status / 系统
  -> status

文档列表 / documents / 文件列表 / 有哪些文档
  -> documents

其他输入
  -> knowledge
```

这样做的好处是：项目在没有模型服务时依然可以演示 LangGraph 工作流，不会因为外部模型不可用而整个 Agent 接口失效。

## 升级后的价值

这次升级后，项目的 LangGraph 部分更像一个真正的工作流 Agent：

- LLM 负责局部语义决策。
- LangGraph 负责整体流程编排。
- 条件边显式表达不同分支。
- 每个工具动作独立成节点。
- state 中保留路由结果和决策原因。

可以用一句话概括：

```text
整体流程由 LangGraph 控制，局部路由决策交给 LLM 完成。
```

这比原来的关键词路由更能体现 LangGraph 的价值，也更适合在面试中讲解。

## Dynamic RAG 图化改造

在进一步升级中，`/api/agent/dynamic-rag` 也从原来的 `create_agent + dynamic_prompt middleware` 改造成了显式 LangGraph 工作流，代码位于 `app/workflows/dynamic_rag.py`。

原来的 Dynamic RAG 流程虽然具备 Query Rewrite、HyDE 和检索上下文注入，但这些步骤隐藏在 `dynamic_prompt` middleware 内部，并且调试信息依赖全局变量 `LAST_REWRITE_INFO`。

改造后，Dynamic RAG 的流程被拆成四个 LangGraph 节点：

```text
rewrite_query
  -> generate_hyde
  -> retrieve_context
  -> generate_answer
  -> END
```

每个节点职责如下：

```text
rewrite_query
  将用户问题改写成更适合向量检索的关键词。

generate_hyde
  基于原问题和关键词生成 HyDE 虚构答案。

retrieve_context
  使用 HyDE 虚构答案作为 retrieval_query 检索 Chroma 知识库。

generate_answer
  将原问题、重写关键词、HyDE、检索上下文一起注入 prompt，调用模型生成最终回答。
```

对应的 state 包含：

```python
original_query
rewritten_query
hyde_answer
retrieval_query
top_k
retrieved_chunks
context
graph_path
output
```

这样做的好处是：

- Dynamic RAG 的每一步都变成显式节点，便于理解和调试。
- 调试信息直接来自 graph state，不再依赖 `LAST_REWRITE_INFO` 全局变量。
- 调试信息包含 `graph_path`，可以看到 `rewrite_query -> generate_hyde -> retrieve_context -> generate_answer -> END`。
- `/api/agent/dynamic-rag` 的前端返回字段保持不变，页面不用重写。
- 项目中两条 Agent 路线都和 LangGraph 结合起来了。

现在可以这样区分两个 Agent 按钮：

```text
LangGraph Agent
  重点展示 LLM 路由决策、条件边和工具节点。

Dynamic RAG(Agentic)
  重点展示 Query Rewrite、HyDE、检索上下文注入和最终回答生成。
```

## Dynamic RAG 上下文充分性门控

本次继续在 Dynamic RAG 中增加了一个轻量 evaluator 节点，用于判断检索回来的上下文是否足够支撑回答。

升级前的 Dynamic RAG 流程是：

```text
rewrite_query
  -> generate_hyde
  -> retrieve_context
  -> generate_answer
  -> END
```

升级后的流程变成：

```text
rewrite_query
  -> generate_hyde
  -> retrieve_context
  -> evaluate_context
  -> generate_answer
  -> END

evaluate_context
  -> insufficient_context
  -> END
```

其中 `evaluate_context` 是一个 LLM structured output 节点，输出固定结构：

```python
class ContextEvaluation(BaseModel):
    sufficient: bool
    reason: str
```

它的职责不是回答问题，而是判断：

```text
当前知识库上下文是否包含回答用户问题所需的直接证据。
```

为了避免 evaluator 变成“模型自己凭常识判断答案”，prompt 中明确约束：

```text
你是 RAG 检索上下文充分性评估器，不是回答者。
即使你自己知道答案，只要上下文没有提供证据，也必须判定为 insufficient。
不要生成答案，不要补充常识，不要猜测。
```

这样设计后，RAG 的边界更清晰：

- 检索器负责从知识库召回候选 chunk。
- evaluator 负责判断这些 chunk 是否足够支撑回答。
- 生成节点只在上下文足够时才生成最终答案。
- 如果上下文不足，则进入 `insufficient_context`，直接返回“当前知识库中没有足够信息回答该问题”。

这不会破坏 RAG 的初衷。相反，它是在检索和生成之间增加一道“证据门控”，避免模型把不相关 chunk 硬塞进答案，也避免用户误以为知识库中真的包含相关内容。

接口 `/api/agent/dynamic-rag` 的 `extra` 现在额外返回：

```json
{
  "context_sufficient": false,
  "context_evaluation_reason": "上下文只提到相关概念，但没有提供回答问题所需的直接证据。"
}
```

前端 Dynamic RAG 调试区也会展示：

```text
上下文是否足够 (context_sufficient)
上下文充分性判断原因 (context_evaluation_reason)
Graph 全程路径 (graph_path)
```

因此，如果一次问答被门控拦截，`graph_path` 会类似：

```text
rewrite_query -> generate_hyde -> retrieve_context -> evaluate_context -> insufficient_context -> END
```

如果上下文足够，`graph_path` 会类似：

```text
rewrite_query -> generate_hyde -> retrieve_context -> evaluate_context -> generate_answer -> END
```

这部分可以作为面试中的 LangGraph 技术亮点来讲：

```text
我在 Dynamic RAG 中把检索后处理设计成一个显式的 LangGraph 条件分支。检索完成后，不是直接把 top-k chunk 交给模型生成答案，而是先用 LLM structured output 做上下文充分性评估。这个评估节点只判断证据是否足够，不负责回答。然后通过 conditional edge 决定进入 generate_answer 还是 insufficient_context。这样既保留了 RAG 以知识库证据为准的原则，也让工作流具备更强的可解释性和抗幻觉能力。
```

## 多 PDF 上传：Orchestrator-Worker + Send API

项目新增了 `/api/documents/upload-batch` 批量上传接口，用于一次上传多个 PDF 或其他支持的文档。

这部分使用 LangGraph 的 Orchestrator-Worker 模式：

```text
orchestrator
  -> Send(upload_worker: file_1)
  -> Send(upload_worker: file_2)
  -> Send(upload_worker: file_3)
  -> ...
  -> summarize_uploads
  -> END
```

核心代码位于：

```text
app/workflows/document_batch.py
```

其中：

```text
dispatch_upload_workers
  Orchestrator，根据上传文件数量动态生成多个 Send。

upload_worker
  Worker，处理单个文件，复用 DocumentService.ingest_upload()。

summarize_uploads
  汇总所有 worker 的成功结果、失败信息和 graph_path。
```

这样设计的好处是：

- 单 PDF 和多 PDF 复用同一套解析、切分、向量化、入库逻辑。
- 多文件数量不固定，适合用 Send 动态生成 worker。
- 每个文件的成功和失败可以单独记录，不会因为某一个文件失败就丢失全部结果。
- graph_path 可以展示本次批量上传经过了哪些 worker 节点。

## 面试表达参考

可以这样描述这次升级：

```text
项目早期版本使用关键词规则判断用户意图，再根据 route 调用不同工具。后来我将 route_intent 节点升级为 LLM structured output 路由，让模型输出固定格式的 route 和 reason；同时把原来的 run_tool 拆成 query_status、query_documents、search_knowledge 三个节点，并通过 add_conditional_edges 显式表达图分支。这样整体流程仍然由 LangGraph 控制，但局部决策由 LLM 完成，既提升了语义理解能力，也增强了工作流的可解释性和可调试性。
```

## 后续可继续优化

后续可以继续做这些升级：

- 在 `search_knowledge` 后增加最终回答生成节点，让 Agent 不只返回检索片段，而是返回自然语言答案。
- 在检索前增加 query rewrite 节点，提高知识库检索效果。
- 在检索后增加文档相关性判断节点，不相关时自动改写问题并重新检索。
- 为 Dynamic RAG 增加相关性阈值判断，避免知识库无关时仍然使用 top-k 结果生成答案。
- 加入 LangGraph checkpoint，让 Agent 状态可以按 `thread_id` 原生持久化。
- 新增一条 `ToolNode + tools_condition` 的 Agent 图，实现真正的 LLM tool calling 循环。
