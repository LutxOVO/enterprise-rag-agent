from app.workflows.agent_router import build_agent_graph
from app.workflows.dynamic_rag import DYNAMIC_RAG_TOP_K, build_dynamic_prompt_agent
from app.storage.database import save_message


class AgentService:
    """Agent 服务：普通工具路由 Agent + Dynamic Prompt RAG Agent。"""

    def __init__(self) -> None:
        self.graph = build_agent_graph()

    def invoke(self, user_input: str, thread_id: str) -> dict:
        result = self.graph.invoke(
            {
                "input": user_input,
                "thread_id": thread_id,
                "route": "",
                "route_reason": "",
                "graph_path": [],
                "output": "",
                "tool_used": "",
            }
        )
        self._save_turn(thread_id, user_input, result["output"])
        return result

    def invoke_dynamic_rag(self, user_input: str, thread_id: str) -> dict:
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
        """Agent 的对话也复用 PostgreSQL 历史表，方便按 thread_id 区分会话。"""
        save_message(thread_id, "user", user_input)
        save_message(thread_id, "assistant", answer)
