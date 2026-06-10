import json
import site
import sysconfig
import warnings
from pathlib import Path
from typing import Any

from langchain_openai import ChatOpenAI

from app.core.config import settings
from app.rag.embeddings import get_embeddings
from app.rag.llm import generate_answer
from app.services.rag_service import RagService


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATASET_PATH = PROJECT_ROOT / "eval_data" / "ragas_eval_dataset.json"
DEFAULT_OUTPUT_PATH = PROJECT_ROOT / "eval_outputs" / "ragas_result.json"
DEFAULT_SAMPLES_PATH = PROJECT_ROOT / "eval_outputs" / "ragas_samples.json"


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


def load_eval_dataset(path: Path) -> list[dict[str, str]]:
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
) -> dict[str, Any]:
    return {
        "dataset_path": str(dataset_path.resolve()),
        "dataset_mtime": dataset_path.stat().st_mtime,
        "top_k": top_k,
        "max_samples": max_samples,
        "sample_count": sample_count,
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
) -> tuple[list[dict[str, Any]], str] | None:
    """Load cached RAGAS samples when response and contexts are already built."""
    if not samples_path.exists():
        return None

    samples = json.loads(samples_path.read_text(encoding="utf-8"))
    meta_path = get_samples_meta_path(samples_path)
    if not meta_path.exists():
        if max_samples is None or len(samples) == max_samples:
            return samples, "used_legacy_cache_without_meta"
        return None

    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    expected_meta = build_samples_meta(dataset_path, top_k, max_samples, len(samples))
    comparable_keys = ["dataset_path", "dataset_mtime", "top_k", "max_samples", "sample_count"]
    if all(meta.get(key) == expected_meta.get(key) for key in comparable_keys):
        return samples, "used_cache"
    return None


async def build_ragas_samples(dataset: list[dict[str, str]], top_k: int) -> list[dict[str, Any]]:
    """调用项目自己的 RAG 流程，为 RAGAS 生成 response 和 retrieved_contexts。"""
    rag_service = RagService()
    samples: list[dict[str, Any]] = []

    for item in dataset:
        question = item["question"]
        results = rag_service.retrieve(question, top_k=top_k)
        context = rag_service.format_context(results)
        answer = await generate_answer(question, context, history_text="")

        samples.append(
            {
                "user_input": question,
                "response": answer,
                "retrieved_contexts": [result.document.page_content for result in results],
                "reference": item["ground_truth"],
            }
        )

    return samples


def save_samples(
    samples: list[dict[str, Any]],
    samples_path: Path,
    dataset_path: Path,
    top_k: int,
    max_samples: int | None,
) -> None:
    """保存 RAGAS 样本和缓存元数据。"""
    samples_path.parent.mkdir(parents=True, exist_ok=True)
    samples_path.write_text(json.dumps(samples, ensure_ascii=False, indent=2), encoding="utf-8")
    save_samples_meta(samples_path, build_samples_meta(dataset_path, top_k, max_samples, len(samples)))


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

    deepseek_llm = ChatOpenAI(
        model=settings.deepseek_chat_model,
        api_key=settings.resolved_deepseek_api_key,
        base_url=settings.deepseek_base_url,
        temperature=0,
    )
    evaluator_llm = LangchainLLMWrapper(deepseek_llm)
    evaluator_embeddings = LangchainEmbeddingsWrapper(get_embeddings())

    return [
        Faithfulness(llm=evaluator_llm),
        AnswerRelevancy(llm=evaluator_llm, embeddings=evaluator_embeddings),
        LLMContextPrecisionWithReference(llm=evaluator_llm, name="context_precision"),
        ContextEntityRecall(llm=evaluator_llm),
        NoiseSensitivity(llm=evaluator_llm),
        LLMContextRecall(llm=evaluator_llm, name="context_recall"),
    ]


def run_ragas_evaluation(samples: list[dict[str, Any]]):
    """运行 RAGAS 评估。这里保持同步执行，便于初学者理解。"""
    ensure_ragas_vertexai_compat()

    from ragas import EvaluationDataset, evaluate

    evaluation_dataset = EvaluationDataset.from_list(samples)
    return evaluate(
        dataset=evaluation_dataset,
        metrics=build_ragas_metrics(),
        raise_exceptions=False,
        show_progress=False,
    )


def save_ragas_result(result: Any, output_path: Path) -> dict[str, float]:
    """保存明细结果，并返回各数值指标的平均分。"""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df = result.to_pandas()
    output_path.write_text(df.to_json(force_ascii=False, orient="records", indent=2), encoding="utf-8")
    df.to_csv(output_path.with_suffix(".csv"), index=False, encoding="utf-8-sig")

    numeric_df = df.select_dtypes(include="number")
    return {
        column: round(float(value), 4)
        for column, value in numeric_df.mean(numeric_only=True).to_dict().items()
    }


async def get_or_build_samples(
    dataset_path: Path,
    samples_path: Path,
    top_k: int,
    max_samples: int | None,
    force_rebuild: bool,
) -> tuple[list[dict[str, Any]], bool, str]:
    """
    统一处理 RAGAS 样本缓存。

    返回值：
    - samples：RAGAS 所需样本；
    - cache_used：是否复用了缓存；
    - cache_status：缓存状态说明，返回给前端展示。
    """
    if not force_rebuild:
        cached = load_cached_samples(samples_path, dataset_path, top_k, max_samples)
        if cached:
            samples, cache_status = cached
            return samples, True, cache_status

    dataset = load_eval_dataset(dataset_path)
    if max_samples:
        dataset = dataset[:max_samples]

    samples = await build_ragas_samples(dataset, top_k=top_k)
    save_samples(samples, samples_path, dataset_path, top_k, max_samples)
    return samples, False, "rebuilt"


async def build_samples_file(
    dataset_path: Path = DEFAULT_DATASET_PATH,
    samples_path: Path = DEFAULT_SAMPLES_PATH,
    top_k: int = 3,
    max_samples: int | None = None,
    force_rebuild: bool = False,
) -> dict[str, Any]:
    samples, cache_used, cache_status = await get_or_build_samples(
        dataset_path=dataset_path,
        samples_path=samples_path,
        top_k=top_k,
        max_samples=max_samples,
        force_rebuild=force_rebuild,
    )

    return {
        "sample_count": len(samples),
        "dataset_path": str(dataset_path),
        "samples_path": str(samples_path),
        "top_k": top_k,
        "max_samples": max_samples,
        "cache_used": cache_used,
        "cache_status": cache_status,
    }


async def run_evaluation_file(
    dataset_path: Path = DEFAULT_DATASET_PATH,
    samples_path: Path = DEFAULT_SAMPLES_PATH,
    output_path: Path = DEFAULT_OUTPUT_PATH,
    top_k: int = 3,
    max_samples: int | None = None,
    force_rebuild: bool = False,
) -> dict[str, Any]:
    samples, cache_used, cache_status = await get_or_build_samples(
        dataset_path=dataset_path,
        samples_path=samples_path,
        top_k=top_k,
        max_samples=max_samples,
        force_rebuild=force_rebuild,
    )

    result = run_ragas_evaluation(samples)
    metrics = save_ragas_result(result, output_path)

    return {
        "sample_count": len(samples),
        "dataset_path": str(dataset_path),
        "samples_path": str(samples_path),
        "result_path": str(output_path),
        "csv_path": str(output_path.with_suffix(".csv")),
        "top_k": top_k,
        "max_samples": max_samples,
        "cache_used": cache_used,
        "cache_status": cache_status,
        "metrics": metrics,
    }
