"""Lifecycle management for the PostgreSQL-backed LangGraph checkpointer."""

import asyncio
import sys
from contextlib import AsyncExitStack

from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver

from app.core.config import settings


# psycopg async connection 在 Windows ProactorEventLoop 上不受支持。
# FastAPI TestClient 和 uvicorn 会在此模块导入后创建事件循环，因此在启动前
# 切换到 SelectorEventLoop；Linux/Docker 不需要这段兼容处理。
if sys.platform == "win32" and hasattr(asyncio, "WindowsSelectorEventLoopPolicy"):
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())


def _psycopg_url(database_url: str) -> str:
    """Convert the SQLAlchemy URL into the libpq URL expected by psycopg."""
    normalized = database_url.strip()
    if normalized.startswith("postgresql+psycopg://"):
        return "postgresql://" + normalized[len("postgresql+psycopg://") :]
    if normalized.startswith("postgresql://"):
        return normalized
    raise RuntimeError(
        "DATABASE_URL must use PostgreSQL, for example "
        "postgresql+psycopg://user:password@localhost:5432/rag."
    )


class AgentCheckpointRuntime:
    """Own the checkpointer connection for the whole FastAPI process."""

    def __init__(self) -> None:
        self._stack: AsyncExitStack | None = None
        self.checkpointer: AsyncPostgresSaver | None = None

    @property
    def ready(self) -> bool:
        return self.checkpointer is not None

    async def start(self) -> AsyncPostgresSaver:
        if self.checkpointer is not None:
            return self.checkpointer

        database_url = settings.database_url.strip()
        if not database_url:
            raise RuntimeError(
                "DATABASE_URL is required before starting the Agent PostgreSQL checkpointer."
            )

        stack = AsyncExitStack()
        try:
            checkpointer = await stack.enter_async_context(
                AsyncPostgresSaver.from_conn_string(_psycopg_url(database_url), pipeline=False)
            )
            await checkpointer.setup()
        except Exception:
            await stack.aclose()
            raise

        self._stack = stack
        self.checkpointer = checkpointer
        return checkpointer

    async def stop(self) -> None:
        stack, self._stack = self._stack, None
        self.checkpointer = None
        if stack is not None:
            await stack.aclose()


agent_checkpoint_runtime = AgentCheckpointRuntime()
