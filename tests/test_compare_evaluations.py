import json

from scripts.compare_evaluations import _experiment, _delta


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False), encoding="utf-8")


def test_compare_summary_reads_metrics_without_calling_models(tmp_path):
    samples = [
        {
            "sample_id": "q1",
            "question_type": "definition",
            "retrieval_metrics": {"hit_at_1": 1.0, "hit_at_3": 1.0, "mrr": 1.0},
            "timings_ms": {"total_ms": 120},
        },
        {
            "sample_id": "q2",
            "question_type": "comparison",
            "retrieval_metrics": {"hit_at_1": 0.0, "hit_at_3": 1.0, "mrr": 0.5},
            "timings_ms": {"total_ms": 180},
        },
    ]
    result = [
        {"faithfulness": 0.8, "answer_relevancy": 0.7, "context_recall": None},
        {"faithfulness": 0.6, "answer_relevancy": 0.9, "context_recall": 0.5},
    ]
    samples_path = tmp_path / "samples.json"
    result_path = tmp_path / "result.json"
    write_json(samples_path, samples)
    write_json(result_path, result)

    summary = _experiment("baseline", samples_path, result_path)

    assert summary["sample_count"] == 2
    assert summary["metrics"]["faithfulness"] == 0.7
    assert summary["metric_null_counts"]["context_recall"] == 1
    assert summary["retrieval_metrics"]["hit_at_3"] == 1.0
    assert summary["by_question_type"]["definition"]["sample_count"] == 1
    assert _delta(summary, {"metrics": {"faithfulness": 0.9}}, "metrics")["faithfulness"] == 0.2
