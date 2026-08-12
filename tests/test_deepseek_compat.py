from langchain_core.messages import ToolMessage

from app.core.deepseek import DeepSeekChatOpenAI


def _model() -> DeepSeekChatOpenAI:
    return DeepSeekChatOpenAI(
        model="deepseek-v4-flash",
        api_key="test-key",
        base_url="https://example.com",
        extra_body={"thinking": {"type": "enabled"}},
    )


def test_deepseek_reasoning_content_is_saved_on_ai_message():
    model = _model()
    result = model._create_chat_result(
        {
            "id": "response-1",
            "model": "deepseek-v4-flash",
            "choices": [
                {
                    "index": 0,
                    "finish_reason": "tool_calls",
                    "message": {
                        "role": "assistant",
                        "content": "",
                        "reasoning_content": "先查询知识库。",
                        "tool_calls": [
                            {
                                "id": "call-1",
                                "type": "function",
                                "function": {
                                    "name": "search_knowledge_base",
                                    "arguments": '{"query":"测试"}',
                                },
                            }
                        ],
                    },
                }
            ],
        }
    )

    message = result.generations[0].message
    assert message.additional_kwargs["reasoning_content"] == "先查询知识库。"


def test_deepseek_reasoning_content_is_sent_back_after_tool_result():
    model = _model()
    payload = model._get_request_payload(
        [
            {
                "role": "assistant",
                "content": "",
                "reasoning_content": "先查询知识库。",
                "tool_calls": [
                    {
                        "id": "call-1",
                        "type": "function",
                        "function": {
                            "name": "search_knowledge_base",
                            "arguments": '{"query":"测试"}',
                        },
                    }
                ],
            },
            ToolMessage(content="检索结果", tool_call_id="call-1"),
        ]
    )

    assert payload["messages"][0]["reasoning_content"] == "先查询知识库。"
