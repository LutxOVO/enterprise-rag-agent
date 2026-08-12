"""DeepSeek 与 LangChain ChatOpenAI 的兼容层。

DeepSeek 思考模式会在 assistant 消息中返回额外的 ``reasoning_content``。
LangChain 的通用 ChatOpenAI 适配器不会自动保存第三方字段，也不会在下一次
请求时回传它；而 DeepSeek 在带工具调用的思考模式中要求该字段完整回传。
这个小适配器只负责保留和回传该字段，不会把思考内容暴露给前端。
"""

from __future__ import annotations

from typing import Any

from langchain_core.messages import AIMessage
from langchain_openai import ChatOpenAI

from app.core.config import settings


class DeepSeekChatOpenAI(ChatOpenAI):
    """保留 DeepSeek thinking mode 的 reasoning_content。"""

    def _get_request_payload(self, input_: Any, *, stop: list[str] | None = None, **kwargs: Any) -> dict[str, Any]:
        payload = super()._get_request_payload(input_, stop=stop, **kwargs)
        request_messages = payload.get("messages")
        if not isinstance(request_messages, list):
            return payload

        # ChatOpenAI 已经完成了标准消息转换；这里只把 DeepSeek 的非标准字段
        # 加回对应的 assistant 消息，避免影响普通 OpenAI-compatible 请求。
        source_messages = self._convert_input(input_).to_messages()
        for source, target in zip(source_messages, request_messages):
            if not isinstance(source, AIMessage) or not isinstance(target, dict):
                continue
            reasoning_content = source.additional_kwargs.get("reasoning_content")
            if reasoning_content is not None:
                target["reasoning_content"] = str(reasoning_content)
        return payload

    def _create_chat_result(self, response: Any, generation_info: dict[str, Any] | None = None):
        result = super()._create_chat_result(response, generation_info)
        response_dict = response if isinstance(response, dict) else response.model_dump()
        choices = response_dict.get("choices") or []

        # 通用 LangChain 解析器会保留 content/tool_calls，但会丢掉
        # DeepSeek 返回的 reasoning_content；把它放入 additional_kwargs，
        # 下一轮 _get_request_payload 就能重新发送给 DeepSeek。
        for generation, choice in zip(result.generations, choices):
            raw_message = choice.get("message", {}) if isinstance(choice, dict) else {}
            reasoning_content = raw_message.get("reasoning_content")
            if reasoning_content is not None and isinstance(generation.message, AIMessage):
                generation.message.additional_kwargs["reasoning_content"] = str(reasoning_content)
        return result


def build_deepseek_chat_model(
    *,
    temperature: float | None = 0,
    streaming: bool = False,
    timeout: float | None = None,
    max_retries: int | None = None,
) -> DeepSeekChatOpenAI:
    """按项目配置创建 DeepSeek 模型，并显式控制 thinking 模式。"""
    if not settings.resolved_deepseek_api_key:
        raise RuntimeError("DeepSeek LLM requires DEEPSEEK_API_KEY in .env.")

    kwargs: dict[str, Any] = {
        "model": settings.deepseek_chat_model,
        "api_key": settings.resolved_deepseek_api_key,
        "base_url": settings.deepseek_base_url,
        "temperature": temperature,
        "streaming": streaming,
        "extra_body": {
            "thinking": {
                "type": settings.deepseek_thinking_type,
            }
        },
    }
    if timeout is not None:
        kwargs["timeout"] = timeout
    if max_retries is not None:
        kwargs["max_retries"] = max_retries
    return DeepSeekChatOpenAI(**kwargs)


__all__ = ["DeepSeekChatOpenAI", "build_deepseek_chat_model"]
