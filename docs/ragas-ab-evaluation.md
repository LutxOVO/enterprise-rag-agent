# RAGAS 两组检索策略对比

本项目用同一份评测集比较两种检索模式。当前默认数据集是围绕 `AI-Agents-in-Depth-zh-CN.pdf` 整理的 `eval_data/ai_agents_in_depth_eval_dataset.json`（60 条）；旧的 `ragas_ab_dataset.json` 仍保留，但不再是默认值。

```text
baseline:
用户问题 -> Chroma 检索 -> DeepSeek 生成回答

hyde_rewrite:
用户问题 -> DeepSeek Query Rewrite -> DeepSeek HyDE
         -> Chroma 检索 -> DeepSeek 生成回答
```

两种模式的最终回答都调用同一个 `generate_answer()`，因此这项实验主要观察检索策略的影响，而不是更换回答 Prompt 的影响。

## 命令行

先用 1 条问题验证：

```powershell
uv run python scripts\evaluate_ragas.py `
  --retrieval-mode baseline `
  --max-samples 1 `
  --force-rebuild

uv run python scripts\evaluate_ragas.py `
  --retrieval-mode hyde_rewrite `
  --max-samples 1 `
  --force-rebuild
```

确认样本内容后，再去掉 `--max-samples 1` 跑完整 60 条：

```powershell
uv run python scripts\evaluate_ragas.py --retrieval-mode baseline --top-k 3
uv run python scripts\evaluate_ragas.py --retrieval-mode hyde_rewrite --top-k 3
```

输出文件按模式分开：

```text
eval_outputs/ragas_samples_baseline.json
eval_outputs/ragas_result_baseline.json
eval_outputs/ragas_samples_hyde_rewrite.json
eval_outputs/ragas_result_hyde_rewrite.json
```

## 样本字段

两组样本都包含 RAGAS 必需字段：

```json
{
  "user_input": "书中给出的 Agent 核心公式是什么？",
  "response": "模型最终回答",
  "retrieved_contexts": ["检索到的 chunk"],
  "reference": "Agent = LLM + 上下文 + 工具"
}
```

`hyde_rewrite` 还会保存调试字段：

```json
{
  "retrieval_mode": "hyde_rewrite",
  "rewritten_query": "打卡 截止时间",
  "hyde_answer": "员工每天需要在 10:00 前完成线上打卡。",
  "retrieval_query": "员工每天需要在 10:00 前完成线上打卡。"
}
```

这些字段不会作为 RAGAS 指标的输入，但可以帮助你解释某一道题为什么检索结果发生变化。

## 如何比较

对两个 `ragas_result_*.csv` 的同名指标求差：

```text
hyde_rewrite 分数 - baseline 分数
```

`context_precision`、`context_recall`、`context_entity_recall`、`faithfulness` 和 `answer_relevancy` 通常越高越好；`noise_sensitivity` 越低越好。

不要只看总平均分，还要看每道问题的差异。HyDE + Query Rewrite 可能改善长问题、口语化问题和术语不一致的问题，但也可能把模型生成的错误假设带入检索，导致某些问题退化。
