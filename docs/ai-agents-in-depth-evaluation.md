# AI Agent 深度理解 PDF 评测集

## 数据来源

本评测集围绕 `AI-Agents-in-Depth-zh-CN.pdf` 编写。PDF 的主题是 AI Agent 的设计原理与工程实践，覆盖 Agent 基础、上下文工程、记忆与 RAG、工具、Coding Agent、评估、持续进化、多模态和多 Agent 协作。

源文件：`C:\Users\15963\Downloads\AI-Agents-in-Depth-zh-CN.pdf`

PDF 约 320 页，评测集先提供 60 条人工整理的问题。每条数据都包含：

- `question`：交给 RAG 系统的问题；
- `ground_truth`：人工整理的标准答案，作为 RAGAS 的 `reference`；
- `source_section`：问题对应的章节；
- `source_page`：PDF 目录中的书内页码，方便回到原文核对；
- `category` 和 `difficulty`：方便按章节或难度分析结果。

文件位置：

```text
eval_data/ai_agents_in_depth_eval_dataset.json
```

## 这份数据集不是做什么的

问题集不会自动写入 Chroma，也不会代替知识库文档。它是评测输入：程序取出问题，向已经入库的 PDF 检索上下文，调用回答模型生成答案，再把生成结果和 `ground_truth` 一起交给 RAGAS 评估。

因此必须先把同一份 PDF 解析并入库，否则评测集和 Chroma 中的知识不匹配，`context_recall`、`context_precision` 和 `faithfulness` 都没有解释价值。

## 推荐运行流程

### 1. 先清理旧知识库

如果 Chroma 里还保留之前的操作系统试题或公司手册，先在网页控制台清理向量库，或者使用项目已有的清理接口。保留的唯一知识库主体应是这份 AI Agent PDF 的 MinerU 解析结果。

### 2. 上传并解析 PDF

启动项目后，通过网页的批量上传区域上传：

```text
C:\Users\15963\Downloads\AI-Agents-in-Depth-zh-CN.pdf
```

PDF 会经过 MinerU 解析、Markdown 切分、Embedding 和 Chroma 入库。确认文档状态为成功，并在“向量库调试”区域看到 chunk 后，再开始评测。

### 3. 先只生成 3 条样本

这一步会调用回答模型，但不会调用 RAGAS Judge，适合先检查检索内容是否来自目标 PDF：

```powershell
cd D:\pycharm项目\RAG
uv run python scripts\evaluate_ragas.py `
  --dataset eval_data\ai_agents_in_depth_eval_dataset.json `
  --retrieval-mode baseline `
  --max-samples 3 `
  --build-only `
  --force-rebuild
```

打开 `eval_outputs/ragas_samples_baseline.json`，重点检查：

- `retrieved_contexts` 是否来自 AI Agent PDF；
- `response` 是否基于上下文回答；
- `reference` 是否是本数据集中的标准答案。

### 4. 跑 baseline 和 hyde_rewrite 两组

两组必须使用同一份数据集、同一个 Chroma 知识库、同一个 `top_k`。第一次运行每组都加 `--force-rebuild`，确保不是复用旧缓存：

```powershell
uv run python scripts\evaluate_ragas.py `
  --dataset eval_data\ai_agents_in_depth_eval_dataset.json `
  --retrieval-mode baseline `
  --top-k 3 `
  --force-rebuild

uv run python scripts\evaluate_ragas.py `
  --dataset eval_data\ai_agents_in_depth_eval_dataset.json `
  --retrieval-mode hyde_rewrite `
  --top-k 3 `
  --force-rebuild
```

默认输出：

```text
eval_outputs/ragas_samples_baseline.json
eval_outputs/ragas_result_baseline.json
eval_outputs/ragas_samples_hyde_rewrite.json
eval_outputs/ragas_result_hyde_rewrite.json
```

`baseline` 直接用原问题检索；`hyde_rewrite` 先让 DeepSeek 改写查询，再生成 HyDE 假答案，用 HyDE 文本检索，最后仍然用原问题生成最终回答。

### 5. 如何比较结果

比较两个 `ragas_result_*.csv` 中的同名指标：

```text
hyde_rewrite 分数 - baseline 分数
```

通常 `faithfulness`、`answer_relevancy`、`context_precision`、`context_entity_recall` 和 `context_recall` 越高越好；`noise_sensitivity` 越低越好。不要只看总平均值，还要回看每道题的上下文和答案，尤其关注跨章节综合题和长问题。

样本文件现在还会保存：

- `retrieved_contexts_metadata`：每个 chunk 的 `document_id`、文件名、标题 metadata、dense/BM25 排名；
- `retrieval_metrics`：按 `source_section` 标签计算的 `hit_at_1`、`hit_at_3` 和 `mrr`；
- `timings_ms`：查询改写、检索、上下文拼接、LLM 和总耗时；
- `category`、`difficulty`、`question_type`：用于分类聚合，不会传给 RAGAS Judge。

`retrieval_mode` 和 `retrieval_strategy` 是两个不同维度：前者比较原问题与 Query Rewrite + HyDE，后者比较纯 dense 与 BM25 + dense 的 RRF。默认文件名仍是 dense 结果；运行 hybrid 时会生成例如：

```text
eval_outputs/ragas_samples_baseline_hybrid.json
eval_outputs/ragas_result_baseline_hybrid.json
```

生成两组结果后运行：

```powershell
uv run python scripts\compare_evaluations.py
```

脚本生成 `eval_outputs/ragas_ab_summary.json` 和 CSV，包含样本数、RAGAS 平均分、空值数、Hit@K、MRR、平均耗时和按 `question_type` 的分组结果。`delta_hyde_minus_baseline` 只是实验差值，不代表统计显著提升；简历中只能填写真实运行结果。

这 60 条数据适合第一轮工程比较。PDF 主题很广，如果以后要做更稳定的结论，可以按章节扩展到每章 10 到 20 条，并对每个模式重复运行 3 到 5 次，报告均值和波动范围。
