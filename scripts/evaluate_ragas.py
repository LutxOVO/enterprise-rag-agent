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
    run_evaluation_file,
)


async def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate the RAG system with RAGAS and DeepSeek.")
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET_PATH)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH)
    parser.add_argument("--samples-output", type=Path, default=DEFAULT_SAMPLES_PATH)
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--force-rebuild", action="store_true", help="Rebuild cached responses and contexts.")
    parser.add_argument("--build-only", action="store_true", help="Only build RAGAS samples, do not call RAGAS.")
    args = parser.parse_args()

    if args.build_only:
        result = await build_samples_file(
            dataset_path=args.dataset,
            samples_path=args.samples_output,
            top_k=args.top_k,
            max_samples=args.max_samples,
            force_rebuild=args.force_rebuild,
        )
        print(
            f"Built {result['sample_count']} samples: {result['samples_path']} "
            f"(cache_used={result['cache_used']}, cache_status={result['cache_status']})"
        )
        return

    result = await run_evaluation_file(
        dataset_path=args.dataset,
        samples_path=args.samples_output,
        output_path=args.output,
        top_k=args.top_k,
        max_samples=args.max_samples,
        force_rebuild=args.force_rebuild,
    )

    print(result["metrics"])
    print(f"cache_used={result['cache_used']}, cache_status={result['cache_status']}")
    print(f"Saved samples to: {result['samples_path']}")
    print(f"Saved RAGAS result to: {result['result_path']}")


if __name__ == "__main__":
    asyncio.run(main())
