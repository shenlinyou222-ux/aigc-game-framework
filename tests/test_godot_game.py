"""Godot 文件命令通道测试：FileCommandGameAdapter 排队写 inbox。"""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from aigc_core import (
    FileCommandGameAdapter,
    GameCommand,
    GameQuery,
)


class FileCommandGameAdapterTests(unittest.TestCase):
    def test_command_queues_jsonl_line(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            inbox = Path(tmp) / "commands" / "inbox.jsonl"
            adapter = FileCommandGameAdapter(inbox_path=inbox, actor="system")
            result = adapter.command(
                GameCommand(
                    command_type="play_music",
                    params={"track": "rain"},
                )
            )
            self.assertTrue(result.ok)
            self.assertEqual(result.code, "queued")
            self.assertTrue(inbox.exists())
            lines = inbox.read_text(encoding="utf-8").strip().splitlines()
            self.assertEqual(len(lines), 1)
            payload = json.loads(lines[0])
            self.assertEqual(payload["command_type"], "play_music")
            self.assertEqual(payload["params"], {"track": "rain"})
            self.assertEqual(payload["actor"], "system")

    def test_command_appends_multiple(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            inbox = Path(tmp) / "inbox.jsonl"
            adapter = FileCommandGameAdapter(inbox_path=inbox)
            adapter.command(GameCommand(command_type="play_music", params={"track": "a"}))
            adapter.command(GameCommand(command_type="play_music", params={"track": "b"}))
            self.assertEqual(len(inbox.read_text(encoding="utf-8").strip().splitlines()), 2)

    def test_query_not_supported(self) -> None:
        adapter = FileCommandGameAdapter(inbox_path=Path(tempfile.gettempdir()) / "nope.jsonl")
        result = adapter.query(GameQuery(query_type="get_world_state"))
        self.assertFalse(result.ok)
        self.assertEqual(result.code, "not_implemented")


if __name__ == "__main__":
    unittest.main()
