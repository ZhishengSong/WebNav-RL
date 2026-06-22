from __future__ import annotations

import unittest

from rollout.parser import ToolCallParseError, parse_tool_call


class ParseToolCallTests(unittest.TestCase):
    def test_parses_valid_call(self) -> None:
        call = parse_tool_call(' <tool_call>{"name":"click","arguments":{"element_id":"x"}}</tool_call> ')
        self.assertEqual(call.name, "click")
        self.assertEqual(call.arguments, {"element_id": "x"})

    def test_rejects_surrounding_prose(self) -> None:
        with self.assertRaisesRegex(ToolCallParseError, "exactly one"):
            parse_tool_call('I will click. <tool_call>{"name":"click","arguments":{}}</tool_call>')

    def test_reports_invalid_json(self) -> None:
        with self.assertRaises(ToolCallParseError) as caught:
            parse_tool_call('<tool_call>{"name":"click"</tool_call>')
        self.assertEqual(caught.exception.code, "invalid_json")

    def test_rejects_invalid_schema(self) -> None:
        with self.assertRaises(ToolCallParseError) as caught:
            parse_tool_call('<tool_call>{"name":"click","arguments":[]}</tool_call>')
        self.assertEqual(caught.exception.code, "invalid_schema")


if __name__ == "__main__":
    unittest.main()
