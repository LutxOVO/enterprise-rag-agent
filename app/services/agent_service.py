from app.agent.graph import build_agent_graph
from app.agent.dynamic_prompt_agent import LAST_REWRITE_INFO, build_dynamic_prompt_agent
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
                "output": "",
                "tool_used": "",
            }
        )
        self._save_turn(thread_id, user_input, result["output"])
        return result

    def invoke_dynamic_rag(self, user_input: str, thread_id: str) -> dict:
        agent = build_dynamic_prompt_agent()
        result = agent.invoke({"messages": [{"role": "user", "content": user_input}]})
        answer = result["messages"][-1].content

        self._save_turn(thread_id, user_input, str(answer))

        return {
            "output": str(answer),
            "tool_used": "dynamic_prompt_retriever",
            "route": "dynamic_rag",
            "original_query": LAST_REWRITE_INFO.get("original_query", user_input),
            "rewritten_query": LAST_REWRITE_INFO.get("rewritten_query", ""),
            "hyde_answer": LAST_REWRITE_INFO.get("hyde_answer", ""),
            "retrieval_query": LAST_REWRITE_INFO.get("retrieval_query", ""),
            "top_k": LAST_REWRITE_INFO.get("top_k", 3),
            "retrieved_chunks": LAST_REWRITE_INFO.get("retrieved_chunks", []),
        }

    def _save_turn(self, thread_id: str, user_input: str, answer: str) -> None:
        """Agent 的对话也复用 SQLite 历史表，方便按 thread_id 区分会话。"""
        save_message(thread_id, "user", user_input)
        save_message(thread_id, "assistant", answer)
