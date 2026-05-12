from __future__ import annotations
import re
import json
from persona.logger import get_logger

logger = get_logger(__name__)


class ActionParser:
    def parse_action(self, text: str) -> str:
        match = re.search(r"<Action>(.*?)</Action>", text, re.DOTALL)
        if not match:
            logger.warning("ActionParser: 未找到 <Action> 标签，原始输出: %r", text)
            return ""

        content = match.group(1).strip()
        if not content or content == "{}":
            return ""

        try:
            json.loads(content)
            return content
        except json.JSONDecodeError as e:
            logger.warning("ActionParser: JSON 解析失败 (%s)，原始内容: %r", e, content)
            return ""
