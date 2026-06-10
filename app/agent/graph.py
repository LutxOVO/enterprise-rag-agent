from typing import Literal, TypedDict

from langgraph.graph import END, StateGraph

from app.agent.tools import query_document_list, query_system_status, search_knowledge_base


class AgentState(TypedDict):
    input: str
    thread_id: str
    route: str
    output: str
    tool_used: str


def route_intent(state: AgentState) -> AgentState:
    """用简单关键词路由到不同工具，便于初学者理解 Agent 决策过程。"""
    text = state["input"].lower()
    if any(word in text for word in ["状态", "status", "系统"]):
        route = "status"
    elif any(word in text for word in ["文档列表", "documents", "文件列表", "有哪些文档"]):
        route = "documents"
    else:
        route = "knowledge"
    return {**state, "route": route}


def run_tool(state: AgentState) -> AgentState:
    route: Literal["status", "documents", "knowledge"] = state["route"]  # type: ignore[assignment]
    if route == "status":
        output = query_system_status.invoke({})
        tool_used = "query_system_status"
    elif route == "documents":
        output = query_document_list.invoke({})
        tool_used = "query_document_list"
    else:
        output = search_knowledge_base.invoke({"query": state["input"], "top_k": 3})
        tool_used = "search_knowledge_base"

    return {**state, "output": output, "tool_used": tool_used}


def build_agent_graph():
    """构建最小 LangGraph：先判断意图，再执行一个工具。"""
    graph = StateGraph(AgentState)
    graph.add_node("route_intent", route_intent)
    graph.add_node("run_tool", run_tool)
    graph.set_entry_point("route_intent")
    graph.add_edge("route_intent", "run_tool")
    graph.add_edge("run_tool", END)
    return graph.compile()
