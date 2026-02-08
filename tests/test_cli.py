"""Tests for CLI module."""

import json

from qwen3_aligner.cli import format_output
from qwen3_aligner.schemas import AlignmentItem


class TestFormatOutput:

    def test_text_format_with_objects(self):
        items = [
            AlignmentItem(text="hello", start_time=0.24, end_time=0.64),
            AlignmentItem(text="world", start_time=0.64, end_time=0.96),
        ]
        result = format_output(items, "text")
        assert "[0.24s - 0.64s] hello" in result
        assert "[0.64s - 0.96s] world" in result

    def test_json_format_with_objects(self):
        items = [
            AlignmentItem(text="hello", start_time=0.24, end_time=0.64),
        ]
        result = format_output(items, "json")
        parsed = json.loads(result)
        assert len(parsed) == 1
        assert parsed[0]["text"] == "hello"
        assert parsed[0]["start_time"] == 0.24

    def test_text_format_with_dicts(self):
        items = [
            {"text": "test", "start_time": 1.0, "end_time": 2.0},
        ]
        result = format_output(items, "text")
        assert "[1.00s - 2.00s] test" in result

    def test_json_format_with_dicts(self):
        items = [
            {"text": "test", "start_time": 1.0, "end_time": 2.0},
        ]
        result = format_output(items, "json")
        parsed = json.loads(result)
        assert parsed[0]["text"] == "test"

    def test_empty_list(self):
        assert format_output([], "text") == ""
        assert json.loads(format_output([], "json")) == []

    def test_unicode_text(self):
        items = [
            AlignmentItem(text="你好", start_time=0.0, end_time=0.5),
        ]
        result = format_output(items, "json")
        parsed = json.loads(result)
        assert parsed[0]["text"] == "你好"
