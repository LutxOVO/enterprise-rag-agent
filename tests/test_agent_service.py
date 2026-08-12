import pytest
from langchain_core.messages import AIMessage, ToolMessage
from langchain_core.tools import tool
from langgraph.checkpoint.memory import InMemorySaver

from app.agent import tools as agent_tools
from app.services.agent_service import AgentConflictError, AgentService


class ScriptedModel:
    """按消息中是否出现工具结果，模拟一个最小的 OpenAI 兼容模型。"""

    def __init__(self, tool_name: str | None = None, calls: int = 1, *, error_after_tool=False):
        self.tool_name = tool_name
        self.remaining_calls = calls
        self.error_after_tool = error_after_tool

    async def ainvoke(self, messages):
        args = (
            {"batch_id": "batch_id", "item_id": "item_id"}
            if self.tool_name == "retry_upload_item"
            else {"value": "initial"}
        )
        saw_tool_result = any(isinstance(message, ToolMessage) for message in messages)
        if self.tool_name and not saw_tool_result:
            return AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": self.tool_name,
                        "args": args,
                        "id": "call-initial",
                        "type": "tool_call",
                    }
                ],
            )
        if saw_tool_result:
            if self.error_after_tool:
                return AIMessage(content="我观察到工具错误，无法继续。")
            if self.remaining_calls > 0:
                self.remaining_calls -= 1
                return AIMessage(
                    content="",
                    tool_calls=[
                        {
                            "name": self.tool_name,
                            "args": args,
                            "id": f"call-{self.remaining_calls}",
                            "type": "tool_call",
                        }
                    ],
                )
            return AIMessage(content="工具链执行完成。")
        return AIMessage(content="普通回答。")


@tool("read_demo")
async def read_demo(value: str) -> dict:
    """测试用只读工具。"""
    return {"ok": True, "message": f"read {value}"}


@tool("search_knowledge_base")
async def fake_search_knowledge_base(query: str, mode: str = "standard", top_k: int = 4) -> dict:
    """测试用知识库工具，验证模型漏调时的安全兜底。"""
    return {
        "ok": True,
        "kind": "knowledge_search",
        "query": query,
        "mode": mode,
        "has_evidence": True,
        "evidence_usable": True,
        "sources": [
            {
                "document_id": "demo-doc",
                "filename": "demo.md",
                "chunk_index": 0,
                "content_preview": "这是来自测试知识库的资料。",
            }
        ],
        "context": "这是来自测试知识库的资料。",
    }


@pytest.mark.anyio
async def test_agent_supports_no_tool_and_continuous_tool_loop():
    direct = AgentService(InMemorySaver(), model=ScriptedModel())
    direct_events = [event async for event in direct.run_stream("你好", "direct-thread")]
    assert [event["event"] for event in direct_events][0] == "run_started"
    assert [event["event"] for event in direct_events][-2:] == ["answer", "done"]
    direct_stages = [event for event in direct_events if event["event"] == "llm_stage"]
    assert [stage["status"] for stage in direct_stages] == ["started", "completed"]
    assert direct_stages[-1]["reasoning_available"] is False
    assert direct_events[-1]["status"] == "completed"

    looping = AgentService(
        InMemorySaver(),
        model=ScriptedModel("read_demo", calls=1),
        tools=[read_demo],
    )
    events = [event async for event in looping.run_stream("连续查询", "loop-thread")]
    event_names = [event["event"] for event in events]
    assert event_names.count("tool_call") == 2
    assert event_names.count("tool_result") == 2
    assert events[-1]["status"] == "completed"
    state = await looping.get_state("loop-thread")
    assert state["tool_call_count"] == 2
    assert len(state["tool_trace"]) == 2
    assert len(state["llm_trace"]) == 3
    assert [item["trace_order"] for item in state["llm_trace"]] == [1, 3, 5]
    assert [item["trace_order"] for item in state["tool_trace"]] == [2, 4]
    assert all("reasoning_content" not in item for item in state["llm_trace"])


@pytest.mark.anyio
async def test_agent_selects_rag_only_when_needed_and_guards_explicit_knowledge_requests():
    class DirectModel:
        async def ainvoke(self, messages):
            # 故意不调用工具：普通请求应直接结束，知识库请求应被图补一次检索。
            if any(isinstance(message, ToolMessage) for message in messages):
                return AIMessage(content="已根据检索资料回答。")
            return AIMessage(content="这是一个不需要知识库的直接回答。")

    casual_service = AgentService(InMemorySaver(), model=DirectModel(), tools=[fake_search_knowledge_base])
    casual_events = [event async for event in casual_service.run_stream("你好", "routing-casual")]
    assert not [event for event in casual_events if event["event"] == "tool_call"]
    assert casual_events[-1]["status"] == "completed"

    knowledge_service = AgentService(InMemorySaver(), model=DirectModel(), tools=[fake_search_knowledge_base])
    knowledge_events = [
        event
        async for event in knowledge_service.run_stream(
            "请根据知识库回答这个问题",
            "routing-knowledge",
        )
    ]
    tool_calls = [event for event in knowledge_events if event["event"] == "tool_call"]
    assert len(tool_calls) == 1
    assert tool_calls[0]["tool_name"] == "search_knowledge_base"
    assert knowledge_events[-1]["status"] == "completed"
    state = await knowledge_service.get_state("routing-knowledge")
    assert state["tool_trace"][0]["name"] == "search_knowledge_base"
    assert state["last_answer"] == "已根据检索资料回答。"


@pytest.mark.anyio
async def test_llm_stage_only_exposes_reasoning_availability():
    class ThinkingModel:
        async def ainvoke(self, messages):
            return AIMessage(
                content="基于模型阶段完成回答",
                additional_kwargs={"reasoning_content": "这是不应出现在轨迹里的原始内容"},
            )

    service = AgentService(InMemorySaver(), model=ThinkingModel())
    events = [event async for event in service.run_stream("测试思考阶段", "thinking-thread")]
    stages = [event for event in events if event["event"] == "llm_stage"]
    assert stages[-1]["reasoning_available"] is True
    assert "原始内容" not in str(stages)
    state = await service.get_state("thinking-thread")
    assert state["llm_trace"][0]["reasoning_available"] is True
    assert "reasoning_content" not in str(state["llm_trace"])


@pytest.mark.anyio
async def test_model_failure_marks_llm_stage_failed():
    class FailingModel:
        async def ainvoke(self, messages):
            raise RuntimeError("模拟模型故障")

    service = AgentService(InMemorySaver(), model=FailingModel())
    events = [event async for event in service.run_stream("测试模型失败", "model-error-thread")]
    failed_stages = [
        event for event in events
        if event["event"] == "llm_stage" and event["status"] == "failed"
    ]
    assert failed_stages
    assert events[-1]["status"] == "failed"
    state = await service.get_state("model-error-thread")
    assert state["status"] == "failed"
    assert state["llm_trace"][0]["status"] == "failed"


@pytest.mark.anyio
async def test_agent_tool_error_is_observed_without_crashing_graph():
    @tool("failing_demo")
    async def failing_demo(value: str) -> dict:
        """测试用失败工具。"""
        raise RuntimeError(f"failure: {value}")

    service = AgentService(
        InMemorySaver(),
        model=ScriptedModel("failing_demo", calls=0, error_after_tool=True),
        tools=[failing_demo],
    )
    events = [event async for event in service.run_stream("调用失败工具", "error-thread")]
    result_events = [event for event in events if event["event"] == "tool_result"]
    assert result_events
    assert result_events[0]["ok"] is False
    assert events[-1]["status"] == "completed"


@pytest.mark.anyio
async def test_agent_stops_at_max_tool_call_limit(monkeypatch):
    original_limit = agent_tools.settings.agent_max_tool_calls
    monkeypatch.setattr(agent_tools.settings, "agent_max_tool_calls", 1)
    import app.agent.graph as graph_module

    monkeypatch.setattr(graph_module.settings, "agent_max_tool_calls", 1)
    try:
        service = AgentService(
            InMemorySaver(),
            model=ScriptedModel("read_demo", calls=5),
            tools=[read_demo],
        )
        events = [event async for event in service.run_stream("不要无限调用", "limit-thread")]
        assert events[-1]["status"] == "completed"
        assert any("最多" in event.get("answer", "") for event in events)
        assert len([event for event in events if event["event"] == "tool_call"]) == 1
    finally:
        agent_tools.settings.agent_max_tool_calls = original_limit
        graph_module.settings.agent_max_tool_calls = original_limit


@pytest.mark.anyio
async def test_write_tool_has_zero_side_effect_before_approval_and_runs_once(monkeypatch):
    executed: list[tuple[str, str]] = []

    async def fake_retry(batch_id: str, item_id: str):
        executed.append((batch_id, item_id))
        return {"batch_id": batch_id, "item_id": item_id, "status": "completed"}

    monkeypatch.setattr(agent_tools, "retry_batch_item", fake_retry)
    model = ScriptedModel("retry_upload_item", calls=0)
    service = AgentService(
        InMemorySaver(),
        model=model,
        tools=[agent_tools.retry_upload_item],
    )

    first = [event async for event in service.run_stream("重试失败文件", "approval-thread")]
    assert first[-1]["status"] == "awaiting_approval"
    assert executed == []
    state = await service.get_state("approval-thread")
    pending = state["pending_approval"]
    assert pending["tool_name"] == "retry_upload_item"

    resumed = [
        event
        async for event in service.resume_stream(
            "approval-thread",
            pending["approval_id"],
            "approve",
        )
    ]
    assert resumed[-1]["status"] == "completed"
    assert executed == [("batch_id", "item_id")]

    with pytest.raises(AgentConflictError) as error:
        await service.validate_resume("approval-thread", pending["approval_id"], "approve")
    assert error.value.status_code == 409


@pytest.mark.anyio
async def test_rejected_write_never_executes(monkeypatch):
    executed: list[str] = []

    async def fake_retry(batch_id: str, item_id: str):
        executed.append(item_id)
        return {"status": "completed"}

    monkeypatch.setattr(agent_tools, "retry_batch_item", fake_retry)
    service = AgentService(
        InMemorySaver(),
        model=ScriptedModel("retry_upload_item", calls=0),
        tools=[agent_tools.retry_upload_item],
    )
    first = [event async for event in service.run_stream("不要重试", "reject-thread")]
    pending = (await service.get_state("reject-thread"))["pending_approval"]
    assert first[-1]["status"] == "awaiting_approval"

    resumed = [
        event
        async for event in service.resume_stream(
            "reject-thread",
            pending["approval_id"],
            "reject",
            "用户拒绝",
        )
    ]
    assert resumed[-1]["status"] == "completed"
    assert executed == []


@pytest.mark.anyio
async def test_delete_thread_removes_checkpoint_and_is_idempotent():
    service = AgentService(InMemorySaver(), model=ScriptedModel())

    events = [event async for event in service.run_stream("建立可删除会话", "delete-thread")]
    assert events[-1]["status"] == "completed"

    deleted = await service.delete_thread("delete-thread")
    assert deleted["deleted"] is True
    assert await service.checkpointer.aget_tuple(service._config("delete-thread")) is None

    repeated = await service.delete_thread("delete-thread")
    assert repeated["deleted"] is True


@pytest.mark.anyio
async def test_delete_thread_rejects_running_and_waiting_approval():
    service = AgentService(InMemorySaver(), model=ScriptedModel())
    lock = await service._lock_for("running-delete")
    await lock.acquire()
    try:
        with pytest.raises(AgentConflictError) as error:
            await service.delete_thread("running-delete")
        assert error.value.status_code == 409
    finally:
        lock.release()

    approval_service = AgentService(
        InMemorySaver(),
        model=ScriptedModel("retry_upload_item", calls=0),
        tools=[agent_tools.retry_upload_item],
    )
    first = [event async for event in approval_service.run_stream("等待审批后删除", "approval-delete")]
    assert first[-1]["status"] == "awaiting_approval"
    with pytest.raises(AgentConflictError) as error:
        await approval_service.delete_thread("approval-delete")
    assert error.value.status_code == 409
