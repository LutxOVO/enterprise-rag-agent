# 简历项目经历：企业知识库 RAG Agent 问答系统

## 推荐写法

**企业知识库 RAG Agent 问答系统｜Python / FastAPI / LangChain / LangGraph / MinerU API / Qwen Embedding / Chroma / RAGAS**

- 面向企业文档问答场景，搭建“文档上传 -> MinerU 解析 -> Markdown 结构化切分 -> Qwen Embedding -> Chroma 检索 -> DeepSeek 生成”的完整 RAG 流程，支持 PDF、Office、图片、HTML、Markdown 和文本文件。
- 针对复杂文档保留 Markdown 标题 metadata 并进行递归二次切分；实现 Dense 检索与 BM25 + Dense 的 RRF 混合检索，返回 `document_id`、文件名、chunk、score 等来源信息，支持按文档过滤和答案溯源。
- 面向批量导入实现 SHA-256 内容幂等去重、文件级状态机、部分失败隔离、补偿清理和失败重试；增加单文档删除、原文件重建索引、SQLite/Chroma 状态一致性处理和 Docker Compose 持久化部署。
- 使用 LangGraph 编排轻量工具路由和 Dynamic RAG 工作流，在回答前完成 Query Rewrite、HyDE、上下文充分性判断和 Prompt Injection 防护；通过 FastAPI 提供普通问答、SSE 流式输出、检索调试和文档管理接口。
- 构建包含 60 条问题和标准答案的评测集，先用同一批 10 条问题进行 baseline 与 Query Rewrite + HyDE 首轮 A/B 评估，记录 Faithfulness、Answer Relevancy、Context Precision、Context Recall、Hit@K、MRR、空值数和阶段耗时；通过 pytest、API 契约测试、smoke test、CI 和结构化请求日志增强可验证性。

## 面试讲解主线

1. **业务目标**：将分散的企业文档转成可检索、可追踪的知识库，回答必须能够回到来源 chunk。
2. **核心链路**：MinerU 负责复杂文档解析，Markdown splitter 保留标题语义，Qwen 负责向量化，Chroma/BM25 负责检索，DeepSeek 负责生成。
3. **技术取舍**：Dense 擅长语义匹配，BM25 擅长编号和专有名词，RRF 解决两个分数空间不可直接比较的问题；HyDE 作为独立实验变量，不和 hybrid 混在同一组 A/B 中。
4. **工程闭环**：批次状态、内容去重、失败重试、补偿清理、删除/重建索引解决“文件上传成功但知识库不一致”的问题。
5. **证据边界**：具体指标只填写真实 `eval_outputs/ragas_ab_summary.json` 的结果，不虚构准确率、用户量或线上收益。

## 项目边界

- 当前定位是本地学习和演示项目，不虚构真实公司经历、用户量或生产收益。
- MinerU、Qwen、DeepSeek 均通过云端 API 调用；SQLite 和 Chroma 适合单机演示，不宣称支持分布式生产部署。
- Agent 部分是可观测的 LangGraph 路由和工具工作流，不表述为完全自主的生产 Agent。
- RAGAS 是离线评估链路，不放入高频用户请求路径。

## 代码与演示

- 项目说明和运行命令：`README.md`
- 评估运行说明：`docs/ai-agents-in-depth-evaluation.md`
- A/B 汇总脚本：`scripts/compare_evaluations.py`
