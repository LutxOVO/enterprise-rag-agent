"""比较 baseline 和 hyde_rewrite 评估结果，不调用任何外部 API。"""

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.services.evaluation_service import RAGAS_METRIC_NAMES  # noqa: E402


def _read_json(path: Path) -> Any:
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _float_or_none(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _metric_column(row: dict[str, Any], metric_name: str) -> Any:
    if metric_name in row:
        return row[metric_name]
    if metric_name == "noise_sensitivity":
        for key, value in row.items():
            if str(key).startswith("noise_sensitivity"):
                return value
    return None


def _metric_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    metrics: dict[str, float] = {}
    null_counts: dict[str, int] = {}
    for metric_name in RAGAS_METRIC_NAMES:
        values = [_float_or_none(_metric_column(row, metric_name)) for row in rows]
        valid = [value for value in values if value is not None]
        metrics[metric_name] = round(sum(valid) / len(valid), 4) if valid else 0.0
        null_counts[metric_name] = len(values) - len(valid)
    return {"metrics": metrics, "metric_null_counts": null_counts}


def _retrieval_summary(samples: list[dict[str, Any]]) -> dict[str, Any]:
    result: dict[str, float | None] = {}
    for metric_name in ("hit_at_1", "hit_at_3", "mrr"):
        values = [
            _float_or_none(sample.get("retrieval_metrics", {}).get(metric_name))
            for sample in samples
        ]
        valid = [value for value in values if value is not None]
        result[metric_name] = round(sum(valid) / len(valid), 4) if valid else None

    durations = [
        _float_or_none(sample.get("timings_ms", {}).get("total_ms"))
        for sample in samples
    ]
    valid_durations = [value for value in durations if value is not None]
    return {
        "metrics": result,
        "average_total_ms": round(sum(valid_durations) / len(valid_durations), 2)
        if valid_durations
        else None,
    }


def _by_question_type(
    samples: list[dict[str, Any]],
    result_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    grouped: dict[str, list[int]] = {}
    for index, sample in enumerate(samples):
        grouped.setdefault(str(sample.get("question_type") or "unknown"), []).append(index)

    output: dict[str, Any] = {}
    for question_type, indices in grouped.items():
        group_samples = [samples[index] for index in indices]
        group_rows = [result_rows[index] for index in indices if index < len(result_rows)]
        retrieval = _retrieval_summary(group_samples)
        ragas = _metric_summary(group_rows) if group_rows else {
            "metrics": {},
            "metric_null_counts": {},
        }
        output[question_type] = {
            "sample_count": len(group_samples),
            "retrieval_metrics": retrieval["metrics"],
            "average_total_ms": retrieval["average_total_ms"],
            **ragas,
        }
    return output


def _experiment(name: str, samples_path: Path, result_path: Path) -> dict[str, Any]:
    samples = _read_json(samples_path)
    result_rows = _read_json(result_path)
    if not isinstance(samples, list) or not isinstance(result_rows, list):
        raise ValueError(f"Expected JSON arrays in {samples_path} and {result_path}.")

    ragas = _metric_summary(result_rows)
    retrieval = _retrieval_summary(samples)
    return {
        "name": name,
        "sample_count": len(samples),
        "result_row_count": len(result_rows),
        "average_total_ms": retrieval["average_total_ms"],
        "metrics": ragas["metrics"],
        "metric_null_counts": ragas["metric_null_counts"],
        "retrieval_metrics": retrieval["metrics"],
        "by_question_type": _by_question_type(samples, result_rows),
        "samples_path": str(samples_path),
        "result_path": str(result_path),
    }


def _delta(left: dict[str, Any], right: dict[str, Any], key: str) -> dict[str, float | None]:
    names = set(left.get(key, {})) | set(right.get(key, {}))
    result: dict[str, float | None] = {}
    for name in sorted(names):
        left_value = _float_or_none(left.get(key, {}).get(name))
        right_value = _float_or_none(right.get(key, {}).get(name))
        result[name] = round(right_value - left_value, 4) if left_value is not None and right_value is not None else None
    return result


def write_flat_csv(summary: dict[str, Any], output_path: Path) -> None:
    rows = []
    baseline = summary["baseline"]
    improved = summary["hyde_rewrite"]
    for group, key in (("ragas", "metrics"), ("retrieval", "retrieval_metrics")):
        names = set(baseline.get(key, {})) | set(improved.get(key, {}))
        for name in sorted(names):
            rows.append(
                {
                    "group": group,
                    "metric": name,
                    "baseline": baseline.get(key, {}).get(name),
                    "hyde_rewrite": improved.get(key, {}).get(name),
                    "delta_hyde_minus_baseline": summary["delta_hyde_minus_baseline"].get(group, {}).get(name),
                }
            )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]) if rows else ["group", "metric"])
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare two cached RAGAS experiments.")
    parser.add_argument("--baseline-samples", type=Path, default=PROJECT_ROOT / "eval_outputs/ragas_samples_baseline.json")
    parser.add_argument("--baseline-result", type=Path, default=PROJECT_ROOT / "eval_outputs/ragas_result_baseline.json")
    parser.add_argument("--hyde-samples", type=Path, default=PROJECT_ROOT / "eval_outputs/ragas_samples_hyde_rewrite.json")
    parser.add_argument("--hyde-result", type=Path, default=PROJECT_ROOT / "eval_outputs/ragas_result_hyde_rewrite.json")
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "eval_outputs/ragas_ab_summary.json")
    parser.add_argument("--csv-output", type=Path, default=None)
    args = parser.parse_args()

    baseline = _experiment("baseline", args.baseline_samples, args.baseline_result)
    hyde_rewrite = _experiment("hyde_rewrite", args.hyde_samples, args.hyde_result)
    summary = {
        "experiment": "baseline_vs_hyde_rewrite",
        "baseline": baseline,
        "hyde_rewrite": hyde_rewrite,
        "delta_hyde_minus_baseline": {
            "ragas": _delta(baseline, hyde_rewrite, "metrics"),
            "retrieval": _delta(baseline, hyde_rewrite, "retrieval_metrics"),
        },
        "interpretation": {
            "higher_is_better": [
                "faithfulness",
                "answer_relevancy",
                "context_precision",
                "context_entity_recall",
                "context_recall",
                "hit_at_1",
                "hit_at_3",
                "mrr",
            ],
            "lower_is_better": ["noise_sensitivity"],
        },
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    csv_output = args.csv_output or args.output.with_suffix(".csv")
    write_flat_csv(summary, csv_output)

    print("metric                         baseline   hyde_rewrite   delta")
    print("-" * 70)
    for group, key in (("ragas", "metrics"), ("retrieval", "retrieval_metrics")):
        for name in sorted(set(baseline.get(key, {})) | set(hyde_rewrite.get(key, {}))):
            print(
                f"{group}.{name:<24} "
                f"{str(baseline.get(key, {}).get(name)):<10} "
                f"{str(hyde_rewrite.get(key, {}).get(name)):<14} "
                f"{summary['delta_hyde_minus_baseline'][group].get(name)}"
            )
    print(f"Saved summary to: {args.output}")
    print(f"Saved CSV to: {csv_output}")


if __name__ == "__main__":
    main()
