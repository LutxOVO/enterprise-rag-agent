from typing import Literal, TypedDict

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import END, StateGraph
from pydantic import BaseModel, Field

from app.tools.rag_tools import query_document_list, query_system_status, search_knowledge_base
from app.core.config import settings
from app.core.deepseek import build_deepseek_chat_model


RouteName = Literal["status", "documents", "knowledge"]


class AgentState(TypedDict):
    input: str
    thread_id: str
    route: str
    route_reason: str
    graph_path: list[str]
    output: str
    tool_used: str


class RouteDecision(BaseModel):
    route: RouteName = Field(description="Selected workflow route.")
    reason: str = Field(description="Short reason for selecting this route.")


def _build_router_model() -> ChatOpenAI | None:
    provider = settings.chat_provider.lower()
    if provider == "qwen" and settings.resolved_qwen_api_key:
        return ChatOpenAI(
            model=settings.qwen_chat_model,
            api_key=settings.resolved_qwen_api_key,
            base_url=settings.qwen_base_url,
            temperature=0,
        )

    if provider == "deepseek" and settings.resolved_deepseek_api_key:
        return build_deepseek_chat_model(temperature=0)

    if provider == "openai" and settings.openai_api_key:
        return ChatOpenAI(
            model=settings.chat_model,
            api_key=settings.openai_api_key,
            base_url=settings.openai_base_url,
            temperature=0,
        )

    return None


def _keyword_route(user_input: str) -> RouteDecision:
    text = user_input.lower()
    status_keywords = [
        "系统状态",
        "运行状态",
        "应用状态",
        "服务状态",
        "健康检查",
        "status",
        "health",
        "文档数量",
        "chunk 数量",
        "chunk数量",
        "向量库路径",
        "embedding provider",
    ]
    if any(word in text for word in status_keywords):
        return RouteDecision(route="status", reason="命中系统状态相关关键词，进入系统状态查询。")
    if any(word in text for word in ["文档列表", "documents", "文件列表", "有哪些文档"]):
        return RouteDecision(route="documents", reason="命中文档列表相关关键词，进入文档列表查询。")
    return RouteDecision(route="knowledge", reason="未命中系统状态或文档列表意图，进入知识库检索。")


def route_intent(state: AgentState) -> AgentState:
    """Use an LLM to select the workflow route, with keyword fallback for local demos."""
    model = _build_router_model()
    graph_path = [*state.get("graph_path", []), "route_intent"]
    if model is None:
        decision = _keyword_route(state["input"])
        return {**state, "route": decision.route, "route_reason": decision.reason, "graph_path": graph_path}

    router = model.with_structured_output(RouteDecision)
    try:
        decision = router.invoke(
            [
                SystemMessage(
                    content=(
                        "你是 LangGraph 工作流中的路由节点，只负责选择下一步分支。"
                        "必须从下面三个 route 中选择一个：\n\n"
                        "1. status：用户想查询系统运行状态、应用状态、文档数量、chunk 数量、"
                        "embedding provider 或向量库路径。\n"
                        "2. documents：用户想查看已上传文档列表、知识库里有哪些文件、"
                        "已有资料清单。\n"
                        "3. knowledge：用户提出任何需要根据知识库内容检索的问题，"
                        "包括技术概念、公司制度、文档内容、业务知识、学习问题、"
                        "以及所有不属于 status/documents 的普通问答。\n\n"
                        "示例：\n"
                        "- “计算机系统结构是什么”“操作系统是什么”“CPU 流水线是什么”"
                        "都属于 knowledge，因为这是知识内容检索，不是查询本应用运行状态。\n"
                        "- “当前系统状态怎么样”“现在有多少 chunk”“向量库路径是什么”"
                        "才属于 status。\n\n"
                        "重要规则：\n"
                        "- 不要把 knowledge 解释成不查知识库的通用知识回答。\n"
                        "- 选择 knowledge 时，reason 必须明确说明“进入知识库检索”。\n"
                        "- 如果不确定，选择 knowledge。\n"
                        "- reason 用中文，简洁说明路由原因。"
                    )
                ),
                HumanMessage(content=state["input"]),
            ]
        )
    except Exception as exc:
        fallback = _keyword_route(state["input"])
        decision = RouteDecision(
            route=fallback.route,
            reason=f"LLM 路由失败（{type(exc).__name__}），使用 fallback：{fallback.reason}",
        )

    return {**state, "route": decision.route, "route_reason": decision.reason, "graph_path": graph_path}


def route_to_tool_node(state: AgentState) -> RouteName:
    route = state.get("route", "knowledge")
    if route in {"status", "documents", "knowledge"}:
        return route  # type: ignore[return-value]
    return "knowledge"


def query_status_node(state: AgentState) -> AgentState:
    output = query_system_status.invoke({})
    graph_path = [*state.get("graph_path", []), "query_status", "END"]
    return {**state, "graph_path": graph_path, "output": output, "tool_used": "query_system_status"}


def query_documents_node(state: AgentState) -> AgentState:
    output = query_document_list.invoke({})
    graph_path = [*state.get("graph_path", []), "query_documents", "END"]
    return {**state, "graph_path": graph_path, "output": output, "tool_used": "query_document_list"}


def search_knowledge_node(state: AgentState) -> AgentState:
    output = search_knowledge_base.invoke({"query": state["input"], "top_k": 3})
    graph_path = [*state.get("graph_path", []), "search_knowledge", "END"]
    return {**state, "graph_path": graph_path, "output": output, "tool_used": "search_knowledge_base"}


def build_agent_graph():
    """Build a LangGraph workflow with LLM routing and explicit conditional edges."""
    graph = StateGraph(AgentState)
    graph.add_node("route_intent", route_intent)
    graph.add_node("query_status", query_status_node)
    graph.add_node("query_documents", query_documents_node)
    graph.add_node("search_knowledge", search_knowledge_node)

    graph.set_entry_point("route_intent")
    graph.add_conditional_edges(
        "route_intent",
        route_to_tool_node,
        {
            "status": "query_status",
            "documents": "query_documents",
            "knowledge": "search_knowledge",
        },
    )
    graph.add_edge("query_status", END)
    graph.add_edge("query_documents", END)
    graph.add_edge("search_knowledge", END)
    return graph.compile()
