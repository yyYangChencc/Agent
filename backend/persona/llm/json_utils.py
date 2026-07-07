from __future__ import annotations

import json
import re
from typing import Any


def strip_json_fence(raw: str) -> str:
    """去掉模型常见的 Markdown JSON 代码块包裹。"""

    text = str(raw or "").strip()
    if not text.startswith("```"):
        return text
    text = re.sub(r"^```(?:json|JSON)?", "", text).strip()
    if text.endswith("```"):
        text = text[:-3].strip()
    return text


def first_json_object_text(raw: str) -> str:
    """从模型输出中提取第一个合法 JSON object 文本。"""

    text = strip_json_fence(raw)
    decoder = json.JSONDecoder()
    for index, char in enumerate(text):
        if char != "{":
            continue
        try:
            value, end = decoder.raw_decode(text[index:])
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            return text[index : index + end]
    return text


def parse_json_object(raw: str, *, context: str = "LLM output") -> dict[str, Any]:
    """解析 LLM JSON object；失败时抛出带上下文的异常。"""

    text = first_json_object_text(raw)
    try:
        data = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ValueError(f"{context} is not valid JSON: {exc}") from exc
    if not isinstance(data, dict):
        raise ValueError(f"{context} is not a JSON object")
    return data
