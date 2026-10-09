"""AI 上层占位网关测试：只读/写助手方法走 controller 权限。"""

from __future__ import annotations

import unittest

from aigc_core import MainController
from aigc_core.ai_gateway import AILayer


class FakeAI(AILayer):
    async def handle_player_input(self, player_id: str, text: str) -> dict:
        return {"player_id": player_id, "text": text}


class AiGatewayTests(unittest.TestCase):
    def setUp(self) -> None:
        self.controller = MainController()
        self.ai = FakeAI(self.controller)

    def test_read_helper_ok(self) -> None:
        r = self.ai._read_game("get_inventory", {"player_id": "player_001"})
        self.assertTrue(r.ok)

    def test_write_helper_denied_for_create_item(self) -> None:
        # assistant(AI 上层) 不能 create_item
        r = self.ai._write_game(
            "create_item",
            {"item_id": "test", "name": "test", "icon_path": "res://x.png"},
        )
        self.assertFalse(r.ok)
        self.assertEqual(r.code, "denied")

    def test_write_helper_ok_send_message(self) -> None:
        r = self.ai._write_game("send_message", {"player_id": "player_001", "text": "hi"})
        self.assertTrue(r.ok)


if __name__ == "__main__":
    unittest.main()
