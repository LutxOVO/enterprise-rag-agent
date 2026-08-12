"""企业知识运营 Agent 服务层。

这里负责把 LangGraph 的 checkpoint、SSE 事件和审批协议接到 FastAPI。业务工具
仍然放在 ``app.agent.tools``，旧的 Dynamic RAG 实验接口也继续保留。
"""

from __future__ import annotations

import asyncio
import json
import uuid
from collections.abc import AsyncIterator
from datetime import datetime, timezone
from typing import Any

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langgraph.types import Command

from app.agent.graph import build_agent_graph, model_is_configured
from app.agent.tools import WRITE_TOOL_NAMES
from app.core.config import settings
from app.workflows.dynamic_rag import DYNAMIC_RAG_TOP_K, build_dynamic_prompt_agent
from app.storage.database import save_message


class AgentServiceError(RuntimeError):
    """可由 API 转换为带 HTTP 状态码的 Agent 错误。"""

    status_code = 500

    def __init__(self, detail: str, *, payload: dict[str, Any] | None = None) -> None:
        super().__init__(detail)
        self.detail = detail
        self.payload = payload or {}


class AgentUnavailableError(AgentServiceError):
    status_code = 503


class AgentConflictError(AgentServiceError):
    status_code = 409


def _json_safe(value: Any) -> Any:
    if hasattr(value, "model_dump"):
        return _json_safe(value.model_dump(mode="json"))
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if hasattr(value, "isoformat"):
        return value.isoformat()
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def _parse_json_content(content: Any) -> dict[str, Any]:
    if isinstance(content, dict):
        return _json_safe(content)
    if isinstance(content, str):
        try:
            value = json.loads(content)
        except json.JSONDecodeError:
            return {"ok": False, "error": content[:1000]}
        return value if isinstance(value, dict) else {"value": _json_safe(value)}
    return {"value": _json_safe(content)}


def _redact_context(value: Any) -> Any:
    """轨迹和 SSE 不重复发送整段上下文，来源列表仍完整保留。"""
    if isinstance(value, dict):
        return {
            str(key): _redact_context(item)
            for key, item in value.items()
            if key not in {"context", "hyde_answer"}
        }
    if isinstance(value, list):
        return [_redact_context(item) for item in value]
    return _json_safe(value)


def _message_tool_calls(message: Any) -> list[dict[str, Any]]:
    if not isinstance(message, AIMessage):
        return []
    return [
        {
            "call_id": str(call.get("id") or ""),
            "tool_name": str(call.get("name") or ""),
            "args": _json_safe(call.get("args") or {}),
        }
        for call in message.tool_calls
    ]


def _message_text(message: Any) -> str:
    content = getattr(message, "content", "")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(
            str(item.get("text", "")) if isinstance(item, dict) else str(item)
            for item in content
        )
    return str(content)


class AgentService:
    """管理单实例 Agent 图，并保证同一 thread_id 串行执行。"""

    def __init__(
        self,
        checkpointer: Any = None,
        *,
        model: Any = None,
        tools: list[Any] | None = None,
    ) -> None:
        self.checkpointer = checkpointer
        self.graph: Any | None = None
        self._test_model = model
        self._test_tools = tools
        self._thread_locks: dict[str, asyncio.Lock] = {}
        self._thread_locks_guard = asyncio.Lock()
        if checkpointer is not None:
            self.configure(checkpointer, model=model, tools=tools)

    def configure(
        self,
        checkpointer: Any,
        *,
        model: Any = None,
        tools: list[Any] | None = None,
    ) -> None:
        """应用启动时调用一次，创建并编译带 PostgreSQL checkpointer 的图。"""
        if self.graph is not None and self.checkpointer is checkpointer:
            return
        self.checkpointer = checkpointer
        self._test_model = model
        self._test_tools = tools
        if model is not None:
            self.graph = build_agent_graph(checkpointer=checkpointer, model=model, tools=tools)
        else:
            self.graph = build_agent_graph(checkpointer=checkpointer) if model_is_configured() else None

    def stop(self) -> None:
        """释放图引用；checkpointer 的连接由 AgentCheckpointRuntime 负责关闭。"""
        self.graph = None
        self.checkpointer = None
        self._test_model = None
        self._test_tools = None
        self._thread_locks.clear()

    @property
    def checkpointer_ready(self) -> bool:
        return self.checkpointer is not None

    @property
    def ready(self) -> bool:
        return self.checkpointer is not None and self.graph is not None

    def _require_graph(self) -> Any:
        if self.checkpointer is None:
            raise AgentUnavailableError("Agent PostgreSQL checkpointer is not ready.")
        if self.graph is None:
            raise AgentUnavailableError(
                "Agent model is not configured. Set the API key for CHAT_PROVIDER "
                f"{settings.chat_provider!r} in .env and restart the service."
            )
        return self.graph

    async def _lock_for(self, thread_id: str) -> asyncio.Lock:
        async with self._thread_locks_guard:
            return self._thread_locks.setdefault(thread_id, asyncio.Lock())

    @staticmethod
    def _config(thread_id: str) -> dict[str, Any]:
        return {"configurable": {"thread_id": thread_id}}

    @staticmethod
    def _pending_from_snapshot(snapshot: Any) -> dict[str, Any] | None:
        values = getattr(snapshot, "values", {}) or {}
        pending = values.get("pending_approval")
        if isinstance(pending, dict):
            return _json_safe(pending)
        for interrupt_item in getattr(snapshot, "interrupts", ()) or ():
            value = getattr(interrupt_item, "value", None)
            if isinstance(value, dict):
                return _json_safe(value)
        return None

    @staticmethod
    def _is_expired(pending: dict[str, Any]) -> bool:
        expires_at = pending.get("expires_at")
        if not expires_at:
            return False
        try:
            expiry = datetime.fromisoformat(str(expires_at).replace("Z", "+00:00"))
        except ValueError:
            return False
        if expiry.tzinfo is None:
            expiry = expiry.replace(tzinfo=timezone.utc)
        return expiry <= datetime.now(timezone.utc)

    async def _snapshot(self, thread_id: str) -> Any:
        graph = self._require_graph()
        return await graph.aget_state(self._config(thread_id))

    async def _check_new_run(self, graph: Any, thread_id: str) -> None:
        snapshot = await graph.aget_state(self._config(thread_id))
        pending = self._pending_from_snapshot(snapshot)
        if pending:
            raise AgentConflictError(
                "该线程正在等待审批，请先批准或拒绝当前操作。",
                payload={"pending_approval": pending, "status": "awaiting_approval"},
            )
        if getattr(snapshot, "next", ()) and (getattr(snapshot, "values", {}) or {}).get("status") == "running":
            raise AgentConflictError("该线程仍有未完成的 Agent 运行，请稍后再试。")

    async def validate_new_run(self, thread_id: str) -> None:
        """SSE 响应创建前做一次冲突检查；执行时还会在锁内再次检查。"""
        graph = self._require_graph()
        lock = await self._lock_for(thread_id)
        async with lock:
            await self._check_new_run(graph, thread_id)

    async def reserve_new_run(self, thread_id: str) -> asyncio.Lock:
        """为 SSE 请求预占 thread 锁，直到对应流结束才释放。"""
        graph = self._require_graph()
        lock = await self._lock_for(thread_id)
        await lock.acquire()
        try:
            await self._check_new_run(graph, thread_id)
        except Exception:
            lock.release()
            raise
        return lock

    async def validate_resume(self, thread_id: str, approval_id: str, decision: str) -> None:
        """在创建 SSE 响应前验证审批，避免把 409 变成半开连接。"""
        graph = self._require_graph()
        lock = await self._lock_for(thread_id)
        async with lock:
            await self._check_resume(graph, thread_id, approval_id, decision)

    async def _check_resume(
        self,
        graph: Any,
        thread_id: str,
        approval_id: str,
        decision: str,
    ) -> None:
        if decision not in {"approve", "reject"}:
            raise AgentConflictError("decision 只能是 approve 或 reject。")
        snapshot = await graph.aget_state(self._config(thread_id))
        pending = self._pending_from_snapshot(snapshot)
        if not pending:
            raise AgentConflictError("当前线程没有等待中的审批，可能已处理或已过期。")
        if pending.get("approval_id") != approval_id:
            raise AgentConflictError(
                "approval_id 与当前待审批操作不匹配。",
                payload={"pending_approval": pending, "status": "awaiting_approval"},
            )
        if self._is_expired(pending):
            raise AgentConflictError(
                "该审批已过期，系统不会执行写操作，请重新发起任务。",
                payload={"pending_approval": pending, "status": "awaiting_approval"},
            )

    async def reserve_resume(
        self,
        thread_id: str,
        approval_id: str,
        decision: str,
    ) -> asyncio.Lock:
        """预占恢复请求的 thread 锁，使重复审批直接得到 HTTP 409。"""
        graph = self._require_graph()
        lock = await self._lock_for(thread_id)
        await lock.acquire()
        try:
            await self._check_resume(graph, thread_id, approval_id, decision)
        except Exception:
            lock.release()
            raise
        return lock

    async def get_state(self, thread_id: str) -> dict[str, Any]:
        """返回前端恢复工作台所需的状态，不暴露完整消息历史。"""
        graph = self._require_graph()
        snapshot = await graph.aget_state(self._config(thread_id))
        values = getattr(snapshot, "values", {}) or {}
        pending = self._pending_from_snapshot(snapshot)
        status = str(values.get("status") or ("awaiting_approval" if pending else "idle"))
        if pending:
            status = "awaiting_approval"
        trace = _redact_context(values.get("tool_trace", []))
        return {
            "thread_id": thread_id,
            "status": status,
            "run_id": values.get("run_id"),
            "pending_approval": pending,
            "tool_trace": trace,
            "sources": _json_safe(values.get("sources", [])),
            "tool_call_count": int(values.get("tool_call_count", 0) or 0),
            "approval_status": values.get("approval_status", "none"),
            "last_answer": str(values.get("last_answer", "") or ""),
            "last_error": str(values.get("last_error", "") or ""),
            "web_search_enabled": bool(values.get("web_search_enabled", False)),
        }

    async def delete_thread(self, thread_id: str) -> dict[str, Any]:
        """删除 Agent checkpoint；旧 RAG messages 表不属于新 Agent 会话。"""
        if self.checkpointer is None:
            raise AgentUnavailableError("Agent PostgreSQL checkpointer is not ready.")

        lock = await self._lock_for(thread_id)
        if lock.locked():
            raise AgentConflictError(
                "该会话正在运行或等待审批，完成或拒绝当前操作后才能删除。",
                payload={"thread_id": thread_id},
            )

        async with lock:
            snapshot = await self.checkpointer.aget_tuple(self._config(thread_id))
            if snapshot is None:
                return {
                    "thread_id": thread_id,
                    "deleted": True,
                    "message": "会话不存在或已删除。",
                }

            values = getattr(snapshot, "checkpoint", {}) or {}
            channel_values = values.get("channel_values", {}) if isinstance(values, dict) else {}
            pending = channel_values.get("pending_approval")
            status = channel_values.get("status")
            if pending or status == "running":
                raise AgentConflictError(
                    "该会话正在运行或等待审批，完成或拒绝当前操作后才能删除。",
                    payload={
                        "thread_id": thread_id,
                        "status": "awaiting_approval" if pending else "running",
                        "pending_approval": _json_safe(pending) if pending else None,
                    },
                )

            await self.checkpointer.adelete_thread(thread_id)
            return {
                "thread_id": thread_id,
                "deleted": True,
                "message": "会话已删除。",
            }

    async def _mark_pending(self, config: dict[str, Any], pending: dict[str, Any]) -> None:
        # 不指定 as_node，保留当前被 interrupt 的下一节点；重启后仍可 Command(resume=...)。
        await self.graph.aupdate_state(
            config,
            {
                "status": "awaiting_approval",
                "approval_status": "pending",
                "pending_approval": _json_safe(pending),
            },
        )

    async def _mark_run_started(self, config: dict[str, Any], run_id: str) -> None:
        # 同样不指定 as_node；resume 时 update 不会破坏待执行节点。
        await self.graph.aupdate_state(
            config,
            {"run_id": run_id, "status": "running", "last_error": ""},
        )

    async def _mark_failed(self, config: dict[str, Any], error: Exception) -> None:
        try:
            await self.graph.aupdate_state(
                config,
                {
                    "status": "failed",
                    "last_error": f"{type(error).__name__}: {error}"[:1000],
                    "pending_approval": None,
                },
            )
        except Exception:
            # 原始错误比失败状态更新更重要；健康接口和日志仍能暴露原始异常。
            return

    @staticmethod
    def _tool_result_event(message: ToolMessage, tool_names: dict[str, str]) -> dict[str, Any]:
        result = _parse_json_content(message.content)
        tool_name = tool_names.get(str(message.tool_call_id), "unknown_tool")
        if result.get("message"):
            summary = str(result["message"])
        elif result.get("error"):
            summary = str(result["error"])
        elif result.get("status"):
            summary = str(result["status"])
        else:
            summary = "工具已返回结果。"
        return {
            "call_id": str(message.tool_call_id or ""),
            "tool_name": tool_name,
            "ok": bool(result.get("ok", False)),
            "status": result.get("status", "completed"),
            "summary": summary[:500],
            "result": _redact_context(result),
        }

    def _events_from_update(
        self,
        update: dict[str, Any],
        tool_names: dict[str, str],
    ) -> list[dict[str, Any]]:
        data = update.get("data", update)
        events: list[dict[str, Any]] = []
        if "__interrupt__" in data:
            interrupts = data.get("__interrupt__") or ()
            for item in interrupts:
                value = getattr(item, "value", item)
                if isinstance(value, dict):
                    events.append({"event": "approval_required", **_json_safe(value)})
            return events

        for node, payload in data.items():
            if not isinstance(payload, dict):
                continue
            for message in payload.get("messages", []) or []:
                calls = _message_tool_calls(message)
                for call in calls:
                    tool_names[call["call_id"]] = call["tool_name"]
                    events.append(
                        {
                            "event": "tool_call",
                            **call,
                            "requires_approval": call["tool_name"] in WRITE_TOOL_NAMES,
                        }
                    )
                if isinstance(message, ToolMessage):
                    events.append({"event": "tool_result", **self._tool_result_event(message, tool_names)})
            # 最终回答在统一收尾阶段发送，避免把模型的中间消息误当成答案。
            if node in {"finalize", "limit"} and payload.get("last_answer"):
                events.append({"event": "answer", "answer": str(payload["last_answer"])})
        return events

    async def _stream_graph(
        self,
        graph_input: Any,
        thread_id: str,
        run_id: str,
        *,
        is_resume: bool = False,
    ) -> AsyncIterator[dict[str, Any]]:
        graph = self._require_graph()
        config = self._config(thread_id)
        tool_names: dict[str, str] = {}
        approval_emitted = False
        answer_emitted = False
        try:
            # 恢复执行时前一条 AIMessage 已经在 checkpoint 中，先建立 call_id -> name 映射。
            snapshot = await graph.aget_state(config)
            for message in (getattr(snapshot, "values", {}) or {}).get("messages", []) or []:
                for call in _message_tool_calls(message):
                    tool_names[call["call_id"]] = call["tool_name"]

            yield {
                "event": "run_started",
                "run_id": run_id,
                "thread_id": thread_id,
                "resume": is_resume,
            }
            async for update in graph.astream(
                graph_input,
                config=config,
                stream_mode="updates",
                version="v2",
            ):
                for event in self._events_from_update(update, tool_names):
                    if event.get("event") == "approval_required":
                        approval_emitted = True
                    if event.get("event") == "answer":
                        answer_emitted = True
                    yield event

            snapshot = await graph.aget_state(config)
            pending = self._pending_from_snapshot(snapshot)
            if pending:
                await self._mark_pending(config, pending)
                if not approval_emitted:
                    yield {"event": "approval_required", **pending}
                yield {
                    "event": "done",
                    "status": "awaiting_approval",
                    "run_id": run_id,
                    "thread_id": thread_id,
                }
                return

            values = getattr(snapshot, "values", {}) or {}
            status = str(values.get("status") or "completed")
            if status not in {"completed", "failed", "awaiting_approval"}:
                status = "completed"
            if status == "completed" and values.get("last_answer") and not answer_emitted:
                # finalize/limit 通常已经发送过 answer；恢复路径或自定义图可能没有该节点事件。
                yield {
                    "event": "answer",
                    "answer": str(values.get("last_answer", "")),
                    "sources": _json_safe(values.get("sources", [])),
                }
            if status == "failed" and values.get("last_error"):
                yield {"event": "error", "message": str(values["last_error"])}
            yield {
                "event": "done",
                "status": status,
                "run_id": run_id,
                "thread_id": thread_id,
                "sources": _json_safe(values.get("sources", [])),
            }
        except Exception as exc:
            await self._mark_failed(config, exc)
            yield {"event": "error", "message": f"{type(exc).__name__}: {exc}"[:1000]}
            yield {
                "event": "done",
                "status": "failed",
                "run_id": run_id,
                "thread_id": thread_id,
            }

    async def run_stream(
        self,
        user_input: str,
        thread_id: str,
        *,
        web_search_enabled: bool = False,
        reserved_lock: asyncio.Lock | None = None,
    ) -> AsyncIterator[dict[str, Any]]:
        """启动一轮新的 Agent 运行，按固定 SSE 事件协议输出。"""
        graph = self._require_graph()
        lock = reserved_lock or await self._lock_for(thread_id)
        should_release = reserved_lock is not None
        if reserved_lock is None:
            await lock.acquire()
            should_release = True
        try:
            snapshot = await graph.aget_state(self._config(thread_id))
            pending = self._pending_from_snapshot(snapshot)
            if pending:
                raise AgentConflictError(
                    "该线程正在等待审批，请先批准或拒绝当前操作。",
                    payload={"pending_approval": pending, "status": "awaiting_approval"},
                )
            if getattr(snapshot, "next", ()) and (getattr(snapshot, "values", {}) or {}).get("status") == "running":
                raise AgentConflictError("该线程仍有未完成的 Agent 运行，请稍后再试。")

            run_id = uuid.uuid4().hex
            config = self._config(thread_id)
            graph_input = {
                "messages": [HumanMessage(content=user_input)],
                "run_id": run_id,
                "tool_call_count": 0,
                "tool_trace": [],
                "sources": [],
                "status": "running",
                "approval_status": "none",
                "pending_approval": None,
                "last_answer": "",
                "last_error": "",
                "web_search_enabled": bool(web_search_enabled),
            }
            async for event in self._stream_graph(graph_input, thread_id, run_id):
                yield event
        finally:
            if should_release and lock.locked():
                lock.release()

    async def resume_stream(
        self,
        thread_id: str,
        approval_id: str,
        decision: str,
        reason: str | None = None,
        *,
        reserved_lock: asyncio.Lock | None = None,
    ) -> AsyncIterator[dict[str, Any]]:
        """使用同一 thread_id 恢复 interrupt；批准后写工具只执行一次。"""
        graph = self._require_graph()
        if decision not in {"approve", "reject"}:
            raise AgentConflictError("decision 只能是 approve 或 reject。")
        lock = reserved_lock or await self._lock_for(thread_id)
        should_release = reserved_lock is not None
        if reserved_lock is None:
            await lock.acquire()
            should_release = True
        try:
            config = self._config(thread_id)
            snapshot = await graph.aget_state(config)
            pending = self._pending_from_snapshot(snapshot)
            if not pending:
                raise AgentConflictError("当前线程没有等待中的审批，可能已处理或已过期。")
            if pending.get("approval_id") != approval_id:
                raise AgentConflictError(
                    "approval_id 与当前待审批操作不匹配。",
                    payload={"pending_approval": pending, "status": "awaiting_approval"},
                )
            if self._is_expired(pending):
                raise AgentConflictError(
                    "该审批已过期，系统不会执行写操作，请重新发起任务。",
                    payload={"pending_approval": pending, "status": "awaiting_approval"},
                )

            run_id = uuid.uuid4().hex
            await self._mark_run_started(config, run_id)
            resume_payload = {
                "approval_id": approval_id,
                "decision": decision,
                "reason": reason or "",
            }
            async for event in self._stream_graph(
                Command(resume=resume_payload),
                thread_id,
                run_id,
                is_resume=True,
            ):
                yield event
        finally:
            if should_release and lock.locked():
                lock.release()

    async def invoke(
        self,
        user_input: str,
        thread_id: str,
        *,
        web_search_enabled: bool = False,
    ) -> dict[str, Any]:
        """兼容旧 /api/agent/invoke，内部仍使用新 checkpoint 图。"""
        await self.validate_new_run(thread_id)
        events = [
            event
            async for event in self.run_stream(
                user_input,
                thread_id,
                web_search_enabled=web_search_enabled,
            )
        ]
        done = next((event for event in reversed(events) if event.get("event") == "done"), {})
        state = await self.get_state(thread_id)
        if done.get("status") == "awaiting_approval":
            raise AgentConflictError(
                "该操作需要审批后才能继续。",
                payload={"pending_approval": state.get("pending_approval"), "status": done.get("status")},
            )
        trace = state.get("tool_trace") or []
        return {
            "output": state.get("last_answer", ""),
            "tool_used": trace[-1].get("name", "") if trace else "none",
            "route": "tool_loop" if trace else "direct_answer",
            "route_reason": "model_tool_selection",
            "graph_path": ["agent", "tools", "agent"] if trace else ["agent", "finalize"],
            "tool_trace": trace,
            "sources": state.get("sources", []),
            "status": done.get("status", "completed"),
        }

    def invoke_dynamic_rag(self, user_input: str, thread_id: str) -> dict:
        """保留旧 Dynamic RAG 实验接口；它不使用 Agent checkpoint。"""
        graph = build_dynamic_prompt_agent()
        result = graph.invoke(
            {
                "original_query": user_input,
                "rewritten_query": "",
                "hyde_answer": "",
                "retrieval_query": "",
                "top_k": DYNAMIC_RAG_TOP_K,
                "retrieved_chunks": [],
                "context": "",
                "context_sufficient": False,
                "context_evaluation_reason": "",
                "graph_path": [],
                "output": "",
            }
        )
        answer = result["output"]
        self._save_turn(thread_id, user_input, str(answer))
        return {
            "output": str(answer),
            "tool_used": "dynamic_rag_langgraph",
            "route": "dynamic_rag_graph",
            "used_rag_tool": True,
            "rag_tool_name": "retrieve_context",
            "graph_path": result.get("graph_path", []),
            "original_query": result.get("original_query", user_input),
            "rewritten_query": result.get("rewritten_query", ""),
            "hyde_answer": result.get("hyde_answer", ""),
            "retrieval_query": result.get("retrieval_query", ""),
            "context_sufficient": result.get("context_sufficient", False),
            "context_evaluation_reason": result.get("context_evaluation_reason", ""),
            "top_k": result.get("top_k", DYNAMIC_RAG_TOP_K),
            "retrieved_chunks": result.get("retrieved_chunks", []),
        }

    def _save_turn(self, thread_id: str, user_input: str, answer: str) -> None:
        """旧接口继续写 messages 表；新 Agent 对话以 checkpoint 为准。"""
        save_message(thread_id, "user", user_input)
        save_message(thread_id, "assistant", answer)


agent_service = AgentService()


__all__ = [
    "AgentConflictError",
    "AgentService",
    "AgentServiceError",
    "AgentUnavailableError",
    "agent_service",
]
