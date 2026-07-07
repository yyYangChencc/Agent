from __future__ import annotations

import unittest
import sys
from pathlib import Path

# 让直接运行单测时也能导入 backend 下的 persona 包。
BACKEND_DIR = Path(__file__).resolve().parent
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from persona.llm.json_utils import parse_json_object


class LLMJsonUtilsTest(unittest.TestCase):
    def test_parse_plain_json_object(self):
        # 纯 JSON 是最理想路径。
        self.assertEqual(parse_json_object('{"score": 0.5}')["score"], 0.5)

    def test_parse_fenced_json_object(self):
        # 兼容模型偶尔包上的 Markdown 代码块。
        payload = parse_json_object('```json\n{"score": 0.25}\n```')
        self.assertEqual(payload["score"], 0.25)

    def test_parse_object_with_extra_text(self):
        # 兼容模型在 JSON 前后输出少量解释文本。
        payload = parse_json_object('说明文字 {"score": -0.4, "reason": "ok"} 结束')
        self.assertEqual(payload["score"], -0.4)

    def test_parse_nested_object(self):
        # 使用 JSONDecoder 避免正则贪婪截断嵌套花括号。
        payload = parse_json_object('prefix {"role_card_delta": {"summary": "x"}, "confidence": 0.8}')
        self.assertEqual(payload["role_card_delta"]["summary"], "x")

    def test_reject_non_object(self):
        with self.assertRaises(ValueError):
            parse_json_object("[1, 2, 3]")


if __name__ == "__main__":
    unittest.main()
