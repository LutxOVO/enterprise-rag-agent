from uuid import uuid4

from langchain_core.messages import AIMessage, ToolMessage
from langchain_core.tools import tool
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from langgraph.types import interrupt
import pytest

from app.agent import checkpoint as _checkpoint_runtime  # noqa: F401  # Windows event-loop compatibility
from app.core.config import settings
from app.services.agent_service import AgentService


class RestartFakeModel:
    async def ainvoke(self, messages):
        if any(isinstance(message, ToolMessage) for message in messages):
            return AIMessage(content="重启后恢复完成")
        return AIMessage(
            content="",
            tool_calls=[
                {
                    "name": "checkpoint_write",
                    "args": {"value": "persisted"},
                    "id": "checkpoint-call-1",
                    "type": "tool_call",
                }
            ],
        )


@pytest.mark.anyio
async def test_postgres_checkpoint_survives_agent_service_recreation():
    executed: list[str] = []

    @tool("checkpoint_write")
    async def checkpoint_write(value: str) -> dict:
        """测试 PostgreSQL checkpoint 恢复的写工具。"""
        decision = interrupt(
            {
                "type": "approval_required",
                "approval_id": "checkpoint-approval",
                "tool_name": "checkpoint_write",
                "args": {"value": value},
                "summary": "测试持久化审批",
                "risk": "medium",
                "expires_at": "2099-01-01T00:00:00+00:00",
            }
        )
        if decision.get("decision") == "approve":
            executed.append(value)
            return {"ok": True, "status": "executed"}
        return {"ok": False, "status": "rejected"}

    thread_id = f"restart-thread-{uuid4().hex}"
    database_url = settings.database_url.replace("postgresql+psycopg://", "postgresql://")
    async with AsyncPostgresSaver.from_conn_string(database_url, pipeline=False) as checkpointer:
        await checkpointer.setup()
        service_before_restart = AgentService(
            checkpointer,
            model=RestartFakeModel(),
            tools=[checkpoint_write],
        )
        first = [
            event
            async for event in service_before_restart.run_stream(
                "执行需要审批的任务",
                thread_id,
            )
        ]
        assert first[-1]["status"] == "awaiting_approval"

        service_after_restart = AgentService(
            checkpointer,
            model=RestartFakeModel(),
            tools=[checkpoint_write],
        )
        state = await service_after_restart.get_state(thread_id)
        assert state["status"] == "awaiting_approval"
        assert state["pending_approval"]["approval_id"] == "checkpoint-approval"
        assert state["llm_trace"]
        assert state["llm_trace"][0]["status"] == "completed"
        assert "reasoning_content" not in str(state["llm_trace"])
        assert executed == []

        resumed = [
            event
            async for event in service_after_restart.resume_stream(
                thread_id,
                "checkpoint-approval",
                "approve",
            )
        ]
        assert resumed[-1]["status"] == "completed"
        assert executed == ["persisted"]
