"""录制的 provider 响应。

结构照抄 OpenAI 兼容接口的真实返回，用于在不触网的前提下验证解析逻辑。
DeepSeek 与 OpenAI 的返回结构一致，故共用同一份录制。
"""

from types import SimpleNamespace

CHAT_RESPONSE = {
    "id": "chatcmpl-recorded-001",
    "object": "chat.completion",
    "model": "gpt-4o-mini",
    "choices": [
        {
            "index": 0,
            "message": {"role": "assistant", "content": "办理学籍证明需携带身份证原件及复印件各一份。"},
            "finish_reason": "stop",
        }
    ],
    "usage": {"prompt_tokens": 42, "completion_tokens": 18, "total_tokens": 60},
}

CHAT_RESPONSE_EMPTY_CONTENT = {
    "id": "chatcmpl-recorded-002",
    "choices": [{"index": 0, "message": {"role": "assistant", "content": ""}, "finish_reason": "stop"}],
}

CHAT_RESPONSE_NO_CHOICES = {"id": "chatcmpl-recorded-003", "choices": []}

EMBEDDING_RESPONSE = {
    "object": "list",
    "model": "text-embedding-3-small",
    "data": [
        {"object": "embedding", "index": 0, "embedding": [0.11, 0.22, 0.33, 0.44]},
        {"object": "embedding", "index": 1, "embedding": [0.55, 0.66, 0.77, 0.88]},
    ],
    "usage": {"prompt_tokens": 8, "total_tokens": 8},
}

# 服务端乱序返回时，必须按 index 归位而不是照单全收
EMBEDDING_RESPONSE_OUT_OF_ORDER = {
    "data": [
        {"object": "embedding", "index": 1, "embedding": [0.55, 0.66, 0.77, 0.88]},
        {"object": "embedding", "index": 0, "embedding": [0.11, 0.22, 0.33, 0.44]},
    ]
}

EMBEDDING_RESPONSE_SHORT = {
    "data": [{"object": "embedding", "index": 0, "embedding": [0.11, 0.22, 0.33, 0.44]}]
}


def as_object(payload):
    """把录制的 JSON 还原成属性访问形式，模拟 SDK 返回的模型对象。"""
    if isinstance(payload, dict):
        return SimpleNamespace(**{key: as_object(value) for key, value in payload.items()})
    if isinstance(payload, list):
        return [as_object(item) for item in payload]
    return payload
