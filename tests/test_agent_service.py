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


@pytest.mark.anyio
async def test_agent_supports_no_tool_and_continuous_tool_loop():
    direct = AgentService(InMemorySaver(), model=ScriptedModel())
    direct_events = [event async for event in direct.run_stream("你好", "direct-thread")]
    assert [event["event"] for event in direct_events] == [
        "run_started",
        "answer",
        "done",
    ]
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
