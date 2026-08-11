import pandas as pd

from app.services import evaluation_service


def test_metric_summary_counts_nan_as_null():
    metrics, null_counts = evaluation_service._summarize_metric_columns(
        pd.DataFrame(
            {
                "faithfulness": [1.0, float("nan")],
                "context_recall": [0.5, None],
            }
        )
    )

    assert metrics["faithfulness"] == 1.0
    assert null_counts["faithfulness"] == 1
    assert metrics["context_recall"] == 0.5
    assert null_counts["context_recall"] == 1


def test_deepseek_answer_relevancy_uses_single_generation(monkeypatch):
    monkeypatch.setattr(evaluation_service.settings, "deepseek_api_key", "test-key")
    monkeypatch.setattr(evaluation_service.settings, "qwen_api_key", "test-key")

    metrics = evaluation_service.build_ragas_metrics()
    answer_relevancy = next(metric for metric in metrics if metric.name == "answer_relevancy")

    assert answer_relevancy.strictness == 1
