import json
import asyncio
import site
import sysconfig
import time
import warnings
from pathlib import Path
from typing import Any

from langchain_openai import ChatOpenAI

from app.core.config import settings
from app.core.deepseek import build_deepseek_chat_model
from app.rag.embeddings import get_embeddings
from app.rag.llm import generate_answer
from app.workflows.dynamic_rag import generate_hyde_answer, rewrite_query_for_retrieval
from app.services.rag_service import RagService


PROJECT_ROOT = Path(__file__).resolve().parents[2]
RETRIEVAL_MODES = ("baseline", "hyde_rewrite")
RETRIEVAL_STRATEGIES = ("dense", "hybrid")
RAGAS_METRIC_NAMES = (
    "faithfulness",
    "answer_relevancy",
    "context_precision",
    "context_entity_recall",
    "noise_sensitivity",
    "context_recall",
)
DEFAULT_DATASET_PATH = PROJECT_ROOT / "eval_data" / "ai_agents_in_depth_eval_dataset.json"


def validate_retrieval_mode(retrieval_mode: str) -> str:
    """限制评估模式，避免缓存和结果文件被错误复用。"""
    if retrieval_mode not in RETRIEVAL_MODES:
        supported = ", ".join(RETRIEVAL_MODES)
        raise ValueError(f"Unsupported retrieval mode: {retrieval_mode}. Choose one of: {supported}.")
    return retrieval_mode


def validate_retrieval_strategy(retrieval_strategy: str) -> str:
    if retrieval_strategy not in RETRIEVAL_STRATEGIES:
        supported = ", ".join(RETRIEVAL_STRATEGIES)
        raise ValueError(
            f"Unsupported retrieval strategy: {retrieval_strategy}. Choose one of: {supported}."
        )
    return retrieval_strategy


def default_samples_path(retrieval_mode: str = "baseline", retrieval_strategy: str = "dense") -> Path:
    mode = validate_retrieval_mode(retrieval_mode)
    strategy = validate_retrieval_strategy(retrieval_strategy)
    suffix = mode if strategy == "dense" else f"{mode}_{strategy}"
    return PROJECT_ROOT / "eval_outputs" / f"ragas_samples_{suffix}.json"


def default_result_path(retrieval_mode: str = "baseline", retrieval_strategy: str = "dense") -> Path:
    mode = validate_retrieval_mode(retrieval_mode)
    strategy = validate_retrieval_strategy(retrieval_strategy)
    suffix = mode if strategy == "dense" else f"{mode}_{strategy}"
    return PROJECT_ROOT / "eval_outputs" / f"ragas_result_{suffix}.json"


DEFAULT_OUTPUT_PATH = default_result_path()
DEFAULT_SAMPLES_PATH = default_samples_path()


def resolve_project_path(path_text: str | None, default_path: Path | None = None) -> Path:
    """把前端传来的相对路径统一转换成项目根目录下的绝对路径。"""
    if not path_text:
        if default_path is None:
            raise ValueError("A path value is required.")
        return default_path

    path = Path(path_text)
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def ensure_ragas_vertexai_compat() -> None:
    """
    RAGAS 0.4.x imports langchain_community.chat_models.vertexai.
    langchain-community 0.4+ no longer ships that module, so we create a tiny
    compatibility shim. The project evaluates with DeepSeek, not VertexAI.
    """
    try:
        import langchain_community.chat_models.vertexai  # noqa: F401
        return
    except ModuleNotFoundError:
        pass

    candidate_dirs = [Path(path) for path in site.getsitepackages()]
    purelib = sysconfig.get_paths().get("purelib")
    if purelib:
        candidate_dirs.append(Path(purelib))

    for base_dir in candidate_dirs:
        chat_models_dir = base_dir / "langchain_community" / "chat_models"
        if chat_models_dir.exists():
            shim_path = chat_models_dir / "vertexai.py"
            shim_path.write_text("class ChatVertexAI:\n    pass\n", encoding="utf-8")
            return


def load_eval_dataset(path: Path) -> list[dict[str, Any]]:
    """读取评估数据集，每条数据必须包含 question 和 ground_truth。"""
    data = json.loads(path.read_text(encoding="utf-8"))
    for index, item in enumerate(data, start=1):
        if "question" not in item or "ground_truth" not in item:
            raise ValueError(f"Dataset item {index} must contain question and ground_truth.")
    return data


def get_samples_meta_path(samples_path: Path) -> Path:
    """缓存元数据文件路径，例如 ragas_samples.json.meta.json。"""
    return samples_path.with_suffix(samples_path.suffix + ".meta.json")


def build_samples_meta(
    dataset_path: Path,
    top_k: int,
    max_samples: int | None,
    sample_count: int,
    retrieval_mode: str,
    retrieval_strategy: str = "dense",
) -> dict[str, Any]:
    retrieval_mode = validate_retrieval_mode(retrieval_mode)
    retrieval_strategy = validate_retrieval_strategy(retrieval_strategy)
    return {
        "dataset_path": str(dataset_path.resolve()),
        "dataset_mtime": dataset_path.stat().st_mtime,
        "top_k": top_k,
        "max_samples": max_samples,
        "sample_count": sample_count,
        "retrieval_mode": retrieval_mode,
        "retrieval_strategy": retrieval_strategy,
    }


def save_samples_meta(samples_path: Path, meta: dict[str, Any]) -> None:
    """保存缓存元数据，用来判断 response 和 contexts 是否还能复用。"""
    meta_path = get_samples_meta_path(samples_path)
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")


def load_cached_samples(
    samples_path: Path,
    dataset_path: Path,
    top_k: int,
    max_samples: int | None,
    retrieval_mode: str,
    retrieval_strategy: str = "dense",
) -> tuple[list[dict[str, Any]], str] | None:
    """Load cached RAGAS samples when response and contexts are already built."""
    retrieval_mode = validate_retrieval_mode(retrieval_mode)
    retrieval_strategy = validate_retrieval_strategy(retrieval_strategy)
    if not samples_path.exists():
        return None

    samples = json.loads(samples_path.read_text(encoding="utf-8"))
    meta_path = get_samples_meta_path(samples_path)
    if not meta_path.exists():
        # 旧版缓存没有模式字段，只能安全地当作普通 baseline 样本。
        if (
            retrieval_mode == "baseline"
            and retrieval_strategy == "dense"
            and (max_samples is None or len(samples) == max_samples)
        ):
            return samples, "used_legacy_cache_without_meta"
        return None

    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    expected_meta = build_samples_meta(
        dataset_path,
        top_k,
        max_samples,
        len(samples),
        retrieval_mode,
        retrieval_strategy,
    )
    comparable_keys = [
        "dataset_path",
        "dataset_mtime",
        "top_k",
        "max_samples",
        "sample_count",
        "retrieval_mode",
        "retrieval_strategy",
    ]
    if all(meta.get(key) == expected_meta.get(key) for key in comparable_keys):
        return samples, "used_cache"
    return None


async def _build_retrieval_query(question: str, retrieval_mode: str) -> tuple[str, dict[str, str]]:
    """生成某种评估模式下的检索词；最终回答仍统一使用原始问题。"""
    retrieval_mode = validate_retrieval_mode(retrieval_mode)
    if retrieval_mode == "baseline":
        return question, {"retrieval_query": question}

    # 这两个调用是同步的 OpenAI-compatible 请求，放入线程避免阻塞 FastAPI 事件循环。
    rewritten_query = await asyncio.to_thread(rewrite_query_for_retrieval, question)
    hyde_answer = await asyncio.to_thread(generate_hyde_answer, question, rewritten_query)
    return hyde_answer, {
        "rewritten_query": rewritten_query,
        "hyde_answer": hyde_answer,
        "retrieval_query": hyde_answer,
    }


async def _retrieve_with_strategy(
    rag_service: RagService,
    query: str,
    top_k: int,
    retrieval_strategy: str,
):
    """把同步检索放入线程；兼容旧测试替身没有新参数的情况。"""
    try:
        return await asyncio.to_thread(
            rag_service.retrieve,
            query,
            top_k,
            retrieval_strategy,
        )
    except TypeError as exc:
        if "retrieval_strategy" not in str(exc) and "positional" not in str(exc):
            raise
        return await asyncio.to_thread(rag_service.retrieve, query, top_k)


def _is_relevant_result(result: Any, item: dict[str, Any]) -> bool | None:
    """用评测集标注的章节和检索 metadata 计算可解释的 Hit@K 标签。"""
    source_section = str(item.get("source_section") or "").strip()
    if not source_section:
        return None
    metadata = result.document.metadata
    header_text = " ".join(
        str(metadata.get(key) or "")
        for key in (
            "Header 1",
            "Header 2",
            "Header 3",
            "header_1",
            "header_2",
            "header_3",
            "h1",
            "h2",
            "h3",
            "h4",
        )
    )
    content = str(result.document.page_content or "")
    return source_section in header_text or source_section in content


def _retrieval_metrics(labels: list[bool | None]) -> dict[str, float | None]:
    known_labels = [label for label in labels if label is not None]
    if not known_labels:
        return {"hit_at_1": None, "hit_at_3": None, "mrr": None}
    first_hit_rank = next(
        (rank for rank, label in enumerate(labels, start=1) if label is True),
        None,
    )
    return {
        "hit_at_1": 1.0 if labels and labels[0] is True else 0.0,
        "hit_at_3": 1.0 if any(label is True for label in labels[:3]) else 0.0,
        "mrr": 1.0 / first_hit_rank if first_hit_rank else 0.0,
    }


async def build_ragas_samples(
    dataset: list[dict[str, Any]],
    top_k: int,
    retrieval_mode: str = "baseline",
    retrieval_strategy: str = "dense",
) -> list[dict[str, Any]]:
    """生成问题、回答、上下文，同时保存来源 metadata 和检索指标。"""
    retrieval_mode = validate_retrieval_mode(retrieval_mode)
    retrieval_strategy = validate_retrieval_strategy(retrieval_strategy)
    rag_service = RagService()
    samples: list[dict[str, Any]] = []

    for index, item in enumerate(dataset, start=1):
        question = item["question"]
        started = time.perf_counter()

        query_started = time.perf_counter()
        retrieval_query, query_details = await _build_retrieval_query(question, retrieval_mode)
        query_ms = int((time.perf_counter() - query_started) * 1000)

        retrieval_started = time.perf_counter()
        results = await _retrieve_with_strategy(
            rag_service,
            retrieval_query,
            top_k,
            retrieval_strategy,
        )
        retrieval_ms = int((time.perf_counter() - retrieval_started) * 1000)

        context_started = time.perf_counter()
        context = rag_service.format_context(results)
        context_ms = int((time.perf_counter() - context_started) * 1000)

        answer_started = time.perf_counter()
        answer = await generate_answer(question, context, history_text="")
        answer_ms = int((time.perf_counter() - answer_started) * 1000)

        labels = [_is_relevant_result(result, item) for result in results]
        retrieval_metrics = _retrieval_metrics(labels)
        retrieved_contexts_metadata = [
            {
                "rank": rank,
                "score": result.score,
                "dense_rank": getattr(result, "dense_rank", None),
                "bm25_rank": getattr(result, "bm25_rank", None),
                "metadata": dict(result.document.metadata),
                "relevant": labels[rank - 1],
            }
            for rank, result in enumerate(results, start=1)
        ]

        samples.append({
            # 这些字段用于后续按问题类型聚合，不会传给 RAGAS 指标输入。
            "sample_id": item.get("id") or f"sample_{index:03d}",
            "category": item.get("category", "unknown"),
            "difficulty": item.get("difficulty", "unknown"),
            "question_type": item.get("question_type", "unknown"),
            "source_section": item.get("source_section", ""),
            "source_page": item.get("source_page"),
            "user_input": question,
            "response": answer,
            "retrieved_contexts": [result.document.page_content for result in results],
            "retrieved_contexts_metadata": retrieved_contexts_metadata,
            "reference": item["ground_truth"],
            "retrieval_mode": retrieval_mode,
            "retrieval_strategy": retrieval_strategy,
            "retrieval_metrics": retrieval_metrics,
            "timings_ms": {
                "query_generation_ms": query_ms,
                "retrieval_ms": retrieval_ms,
                "context_ms": context_ms,
                "llm_ms": answer_ms,
                "total_ms": int((time.perf_counter() - started) * 1000),
            },
            **query_details,
        })

    return samples


def save_samples(
    samples: list[dict[str, Any]],
    samples_path: Path,
    dataset_path: Path,
    top_k: int,
    max_samples: int | None,
    retrieval_mode: str,
    retrieval_strategy: str = "dense",
) -> None:
    """保存 RAGAS 样本和缓存元数据。"""
    samples_path.parent.mkdir(parents=True, exist_ok=True)
    samples_path.write_text(json.dumps(samples, ensure_ascii=False, indent=2), encoding="utf-8")
    save_samples_meta(
        samples_path,
        build_samples_meta(
            dataset_path,
            top_k,
            max_samples,
            len(samples),
            retrieval_mode,
            retrieval_strategy,
        ),
    )


def build_ragas_metrics():
    """构造 RAGAS 指标，使用 DeepSeek 作为 Judge Model。"""
    ensure_ragas_vertexai_compat()
    warnings.filterwarnings("ignore", category=DeprecationWarning)

    from ragas.embeddings import LangchainEmbeddingsWrapper
    from ragas.llms import LangchainLLMWrapper
    from ragas.metrics import (
        AnswerRelevancy,
        ContextEntityRecall,
        Faithfulness,
        LLMContextPrecisionWithReference,
        LLMContextRecall,
        NoiseSensitivity,
    )

    if not settings.resolved_deepseek_api_key:
        raise RuntimeError("Please set DEEPSEEK_API_KEY in .env before running RAGAS evaluation.")

    deepseek_llm = build_deepseek_chat_model(
        temperature=0,
        timeout=60,
        max_retries=0,
    )
    evaluator_llm = LangchainLLMWrapper(deepseek_llm)
    evaluator_embeddings = LangchainEmbeddingsWrapper(get_embeddings())

    return [
        Faithfulness(llm=evaluator_llm),
        # DeepSeek 兼容接口只支持 n=1；RAGAS 默认 strictness=3 会请求三组生成。
        AnswerRelevancy(
            llm=evaluator_llm,
            embeddings=evaluator_embeddings,
            strictness=1,
        ),
        LLMContextPrecisionWithReference(llm=evaluator_llm, name="context_precision"),
        ContextEntityRecall(llm=evaluator_llm),
        NoiseSensitivity(llm=evaluator_llm),
        LLMContextRecall(llm=evaluator_llm, name="context_recall"),
    ]


def run_ragas_evaluation(samples: list[dict[str, Any]]):
    """运行 RAGAS 评估。这里保持同步执行，便于初学者理解。"""
    ensure_ragas_vertexai_compat()

    from ragas import EvaluationDataset, evaluate
    from ragas.run_config import RunConfig

    # RAGAS 只需要这四个标准字段；项目自己的分类、来源和时延字段单独保存在样本文件。
    evaluation_rows = [
        {
            "user_input": sample["user_input"],
            "response": sample["response"],
            "retrieved_contexts": sample["retrieved_contexts"],
            "reference": sample["reference"],
        }
        for sample in samples
    ]
    evaluation_dataset = EvaluationDataset.from_list(evaluation_rows)
    return evaluate(
        dataset=evaluation_dataset,
        metrics=build_ragas_metrics(),
        # 云端 Judge 不适合使用 RAGAS 默认的 16 并发；限制并发并关闭重复重试，
        # 避免单条异常拖住整组实验。
        run_config=RunConfig(timeout=60, max_retries=0, max_wait=5, max_workers=2),
        raise_exceptions=False,
        show_progress=False,
    )


def _canonical_metric_name(column: str) -> str | None:
    if column in RAGAS_METRIC_NAMES:
        return column
    if column.startswith("noise_sensitivity"):
        return "noise_sensitivity"
    return None


def _summarize_metric_columns(df: Any) -> tuple[dict[str, float], dict[str, int]]:
    metrics: dict[str, float] = {}
    null_counts: dict[str, int] = {}
    for column in df.columns:
        canonical_name = _canonical_metric_name(str(column))
        if canonical_name is None:
            continue
        # 先转成 Python list，避免 Pandas 把返回的 None 自动提升成 NaN，
        # 否则 NaN 会绕过 ``value is not None`` 判断并污染平均值。
        numeric_values = [_to_float_or_none(value) for value in df[column].tolist()]
        valid_values = [value for value in numeric_values if value is not None]
        metrics[canonical_name] = round(sum(valid_values) / len(valid_values), 4) if valid_values else 0.0
        null_counts[canonical_name] = len(numeric_values) - len(valid_values)
    return metrics, null_counts


def _to_float_or_none(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if number == number else None


def aggregate_retrieval_metrics(samples: list[dict[str, Any]]) -> dict[str, float | None]:
    """聚合 Hit@1、Hit@3、MRR；None 表示评测集没有可匹配的章节标注。"""
    result: dict[str, float | None] = {}
    for metric_name in ("hit_at_1", "hit_at_3", "mrr"):
        values = [
            sample.get("retrieval_metrics", {}).get(metric_name)
            for sample in samples
        ]
        values = [float(value) for value in values if value is not None]
        result[metric_name] = round(sum(values) / len(values), 4) if values else None
    return result


def _group_retrieval_metrics(samples: list[dict[str, Any]], group_key: str) -> dict[str, Any]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for sample in samples:
        group = str(sample.get(group_key) or "unknown")
        grouped.setdefault(group, []).append(sample)

    output: dict[str, Any] = {}
    for group, group_samples in grouped.items():
        output[group] = {
            "sample_count": len(group_samples),
            "retrieval_metrics": aggregate_retrieval_metrics(group_samples),
        }
    return output


def _group_result_metrics(df: Any, samples: list[dict[str, Any]], group_key: str) -> dict[str, Any]:
    """把 RAGAS 分数和检索分数一起按数据集标签聚合。"""
    grouped: dict[str, list[int]] = {}
    for index, sample in enumerate(samples):
        group = str(sample.get(group_key) or "unknown")
        grouped.setdefault(group, []).append(index)

    output: dict[str, Any] = {}
    for group, indices in grouped.items():
        group_samples = [samples[index] for index in indices]
        valid_indices = [index for index in indices if index < len(df)]
        group_df = df.iloc[valid_indices] if valid_indices else df.iloc[0:0]
        group_metrics, group_null_counts = _summarize_metric_columns(group_df)
        output[group] = {
            "sample_count": len(group_samples),
            "metrics": group_metrics,
            "metric_null_counts": group_null_counts,
            "retrieval_metrics": aggregate_retrieval_metrics(group_samples),
        }
    return output


def save_ragas_result(
    result: Any,
    output_path: Path,
    samples: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """保存明细、CSV 和脱敏汇总，并返回平均分与空值数。"""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df = result.to_pandas()
    if samples:
        for field in ("sample_id", "category", "difficulty", "question_type", "retrieval_mode", "retrieval_strategy"):
            df[field] = [sample.get(field) for sample in samples]
    output_path.write_text(df.to_json(force_ascii=False, orient="records", indent=2), encoding="utf-8")
    df.to_csv(output_path.with_suffix(".csv"), index=False, encoding="utf-8-sig")

    metrics, null_counts = _summarize_metric_columns(df)
    summary = {
        "sample_count": len(samples or df),
        "metrics": metrics,
        "metric_null_counts": null_counts,
        "retrieval_metrics": aggregate_retrieval_metrics(samples or []),
        "by_question_type": _group_result_metrics(df, samples or [], "question_type"),
        "by_category": _group_retrieval_metrics(samples or [], "category"),
    }
    summary_path = output_path.with_suffix(".summary.json")
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


async def get_or_build_samples(
    dataset_path: Path,
    samples_path: Path,
    top_k: int,
    max_samples: int | None,
    force_rebuild: bool,
    retrieval_mode: str = "baseline",
    retrieval_strategy: str = "dense",
) -> tuple[list[dict[str, Any]], bool, str]:
    """
    统一处理 RAGAS 样本缓存。

    返回值：
    - samples：RAGAS 所需样本；
    - cache_used：是否复用了缓存；
    - cache_status：缓存状态说明，返回给前端展示。
    """
    retrieval_mode = validate_retrieval_mode(retrieval_mode)
    retrieval_strategy = validate_retrieval_strategy(retrieval_strategy)
    if not force_rebuild:
        cached = load_cached_samples(
            samples_path,
            dataset_path,
            top_k,
            max_samples,
            retrieval_mode,
            retrieval_strategy,
        )
        if cached:
            samples, cache_status = cached
            return samples, True, cache_status

    dataset = load_eval_dataset(dataset_path)
    if max_samples:
        dataset = dataset[:max_samples]

    samples = await build_ragas_samples(
        dataset,
        top_k=top_k,
        retrieval_mode=retrieval_mode,
        retrieval_strategy=retrieval_strategy,
    )
    save_samples(
        samples,
        samples_path,
        dataset_path,
        top_k,
        max_samples,
        retrieval_mode,
        retrieval_strategy,
    )
    return samples, False, "rebuilt"


async def build_samples_file(
    dataset_path: Path = DEFAULT_DATASET_PATH,
    samples_path: Path = DEFAULT_SAMPLES_PATH,
    top_k: int = 3,
    max_samples: int | None = None,
    force_rebuild: bool = False,
    retrieval_mode: str = "baseline",
    retrieval_strategy: str = "dense",
) -> dict[str, Any]:
    samples, cache_used, cache_status = await get_or_build_samples(
        dataset_path=dataset_path,
        samples_path=samples_path,
        top_k=top_k,
        max_samples=max_samples,
        force_rebuild=force_rebuild,
        retrieval_mode=retrieval_mode,
        retrieval_strategy=retrieval_strategy,
    )

    return {
        "sample_count": len(samples),
        "dataset_path": str(dataset_path),
        "samples_path": str(samples_path),
        "top_k": top_k,
        "max_samples": max_samples,
        "retrieval_mode": retrieval_mode,
        "retrieval_strategy": retrieval_strategy,
        "cache_used": cache_used,
        "cache_status": cache_status,
        "retrieval_metrics": aggregate_retrieval_metrics(samples),
    }


async def run_evaluation_file(
    dataset_path: Path = DEFAULT_DATASET_PATH,
    samples_path: Path = DEFAULT_SAMPLES_PATH,
    output_path: Path = DEFAULT_OUTPUT_PATH,
    top_k: int = 3,
    max_samples: int | None = None,
    force_rebuild: bool = False,
    retrieval_mode: str = "baseline",
    retrieval_strategy: str = "dense",
) -> dict[str, Any]:
    samples, cache_used, cache_status = await get_or_build_samples(
        dataset_path=dataset_path,
        samples_path=samples_path,
        top_k=top_k,
        max_samples=max_samples,
        force_rebuild=force_rebuild,
        retrieval_mode=retrieval_mode,
        retrieval_strategy=retrieval_strategy,
    )

    result = run_ragas_evaluation(samples)
    summary = save_ragas_result(result, output_path, samples=samples)

    return {
        "sample_count": len(samples),
        "dataset_path": str(dataset_path),
        "samples_path": str(samples_path),
        "result_path": str(output_path),
        "csv_path": str(output_path.with_suffix(".csv")),
        "top_k": top_k,
        "max_samples": max_samples,
        "retrieval_mode": retrieval_mode,
        "retrieval_strategy": retrieval_strategy,
        "cache_used": cache_used,
        "cache_status": cache_status,
        "metrics": summary["metrics"],
        "metric_null_counts": summary["metric_null_counts"],
        "retrieval_metrics": summary["retrieval_metrics"],
        "summary_path": str(output_path.with_suffix(".summary.json")),
    }
