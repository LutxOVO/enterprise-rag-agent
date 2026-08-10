import argparse
import asyncio
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.services.evaluation_service import (  # noqa: E402
    DEFAULT_DATASET_PATH,
    DEFAULT_OUTPUT_PATH,
    DEFAULT_SAMPLES_PATH,
    build_samples_file,
    default_result_path,
    default_samples_path,
    run_evaluation_file,
)


async def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate the RAG system with RAGAS and DeepSeek.")
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET_PATH)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH)
    parser.add_argument("--samples-output", type=Path, default=None)
    parser.add_argument(
        "--retrieval-mode",
        choices=["baseline", "hyde_rewrite"],
        default="baseline",
        help="baseline: 原问题直接检索；hyde_rewrite: Query Rewrite + HyDE",
    )
    parser.add_argument(
        "--retrieval-strategy",
        choices=["dense", "hybrid"],
        default="dense",
        help="dense: 纯向量；hybrid: dense + BM25 的 RRF 混合检索",
    )
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--force-rebuild", action="store_true", help="Rebuild cached responses and contexts.")
    parser.add_argument("--build-only", action="store_true", help="Only build RAGAS samples, do not call RAGAS.")
    args = parser.parse_args()
    samples_output = args.samples_output or default_samples_path(
        args.retrieval_mode,
        args.retrieval_strategy,
    )
    output_path = args.output
    if output_path == DEFAULT_OUTPUT_PATH and (
        args.retrieval_mode != "baseline" or args.retrieval_strategy != "dense"
    ):
        output_path = default_result_path(args.retrieval_mode, args.retrieval_strategy)

    if args.build_only:
        result = await build_samples_file(
            dataset_path=args.dataset,
            samples_path=samples_output,
            top_k=args.top_k,
            max_samples=args.max_samples,
            force_rebuild=args.force_rebuild,
            retrieval_mode=args.retrieval_mode,
            retrieval_strategy=args.retrieval_strategy,
        )
        print(
            f"Built {result['sample_count']} samples: {result['samples_path']} "
            f"(cache_used={result['cache_used']}, cache_status={result['cache_status']})"
        )
        return

    result = await run_evaluation_file(
        dataset_path=args.dataset,
        samples_path=samples_output,
        output_path=output_path,
        top_k=args.top_k,
        max_samples=args.max_samples,
        force_rebuild=args.force_rebuild,
        retrieval_mode=args.retrieval_mode,
        retrieval_strategy=args.retrieval_strategy,
    )

    print(result["metrics"])
    print(f"retrieval_metrics={result['retrieval_metrics']}")
    print(f"metric_null_counts={result['metric_null_counts']}")
    print(f"cache_used={result['cache_used']}, cache_status={result['cache_status']}")
    print(f"Saved samples to: {result['samples_path']}")
    print(f"Saved RAGAS result to: {result['result_path']}")
    print(f"Saved summary to: {result['summary_path']}")


if __name__ == "__main__":
    asyncio.run(main())
