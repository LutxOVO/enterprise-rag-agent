from types import SimpleNamespace

import pytest
from langchain_core.documents import Document

from app.services import evaluation_service


class FakeRagService:
    def __init__(self, queries: list[str]) -> None:
        self.queries = queries

    def retrieve(self, query: str, top_k: int):
        self.queries.append(query)
        return [SimpleNamespace(document=Document(page_content=f"context for: {query}"), score=0.1)]

    def format_context(self, results) -> str:
        return "\n\n".join(result.document.page_content for result in results)


@pytest.mark.anyio
async def test_baseline_uses_original_question(monkeypatch):
    queries: list[str] = []
    monkeypatch.setattr(evaluation_service, "RagService", lambda: FakeRagService(queries))

    async def fake_generate_answer(question: str, context: str, history_text: str) -> str:
        return f"answer for {question}"

    monkeypatch.setattr(evaluation_service, "generate_answer", fake_generate_answer)

    samples = await evaluation_service.build_ragas_samples(
        [{"question": "员工几点打卡？", "ground_truth": "10:00 前。"}],
        top_k=3,
        retrieval_mode="baseline",
    )

    assert queries == ["员工几点打卡？"]
    assert samples[0]["retrieval_mode"] == "baseline"
    assert samples[0]["retrieval_query"] == "员工几点打卡？"
    assert "rewritten_query" not in samples[0]
    assert samples[0]["response"] == "answer for 员工几点打卡？"


@pytest.mark.anyio
async def test_hyde_rewrite_runs_rewrite_then_hyde(monkeypatch):
    queries: list[str] = []
    monkeypatch.setattr(evaluation_service, "RagService", lambda: FakeRagService(queries))
    call_order: list[str] = []

    def fake_rewrite(question: str) -> str:
        call_order.append("rewrite")
        return "打卡 截止时间"

    def fake_hyde(question: str, rewritten_query: str) -> str:
        call_order.append("hyde")
        assert rewritten_query == "打卡 截止时间"
        return "员工每天需要在 10:00 前完成线上打卡。"

    async def fake_generate_answer(question: str, context: str, history_text: str) -> str:
        call_order.append("answer")
        return "answer"

    monkeypatch.setattr(evaluation_service, "rewrite_query_for_retrieval", fake_rewrite)
    monkeypatch.setattr(evaluation_service, "generate_hyde_answer", fake_hyde)
    monkeypatch.setattr(evaluation_service, "generate_answer", fake_generate_answer)

    samples = await evaluation_service.build_ragas_samples(
        [{"question": "员工几点打卡？", "ground_truth": "10:00 前。"}],
        top_k=3,
        retrieval_mode="hyde_rewrite",
    )

    assert call_order == ["rewrite", "hyde", "answer"]
    assert queries == ["员工每天需要在 10:00 前完成线上打卡。"]
    assert samples[0]["retrieval_mode"] == "hyde_rewrite"
    assert samples[0]["rewritten_query"] == "打卡 截止时间"
    assert samples[0]["hyde_answer"] == "员工每天需要在 10:00 前完成线上打卡。"
    assert samples[0]["retrieval_query"] == samples[0]["hyde_answer"]


def test_invalid_retrieval_mode_is_rejected():
    with pytest.raises(ValueError, match="Unsupported retrieval mode"):
        evaluation_service.validate_retrieval_mode("unknown")
