"""FastAPI HTTP 接口测试（AI/客户端同用的命令与向量入口）。"""

from __future__ import annotations

import unittest

from fastapi.testclient import TestClient

from aigc_core.api import create_app
from aigc_core.models import ActorRole, GameCommand, GameEvent, GameLogRecord, WriterRole


def _client() -> TestClient:
    return TestClient(create_app(persist=False))


class ApiHealthTests(unittest.TestCase):
    def test_health(self) -> None:
        resp = _client().get("/health")
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(resp.json()["status"], "ok")


class ApiEventAndMemoryTests(unittest.TestCase):
    def test_write_event_and_query_memory(self) -> None:
        client = _client()
        ev = {
            "event_id": "ev-1",
            "event_type": "player_chat",
            "source_type": "系统",
            "location": "farm",
            "entity_id": "player_001",
            "payload": {"text": "hi"},
        }
        r = client.post("/v1/events", json={"event": ev, "writer": "sensor"})
        self.assertEqual(r.status_code, 200)
        self.assertTrue(r.json()["ok"])

        r = client.post(
            "/v1/memory/query",
            json={"query": {"event_type": "player_chat", "limit": 10}},
        )
        self.assertEqual(r.status_code, 200)
        items = r.json()["items"]
        self.assertEqual(len(items), 1)
        self.assertEqual(items[0]["event_id"], "ev-1")


class ApiLogTests(unittest.TestCase):
    def test_write_and_query_log(self) -> None:
        client = _client()
        rec = {
            "event_id": "log-1",
            "event_type": "player.move",
            "time": "t",
            "unix_time": 1.0,
            "data": {"moving": True},
            "semantic_vector": [1.0, 0.0, 0.0],
        }
        r = client.post("/v1/logs", json={"record": rec})
        self.assertEqual(r.status_code, 200)
        self.assertTrue(r.json()["ok"])

        q = client.post(
            "/v1/logs/query",
            json={"query": {"query_vector": [0.9, 0.0, 0.0], "limit": 5}},
        )
        self.assertEqual(q.status_code, 200)
        items = q.json()["items"]
        self.assertEqual(len(items), 1)
        self.assertEqual(items[0]["record"]["event_id"], "log-1")
        self.assertGreater(items[0]["score"], 0.9)


class ApiGameTests(unittest.TestCase):
    def test_game_query_world_state(self) -> None:
        r = _client().post(
            "/v1/game/query",
            json={
                "query": {"query_type": "get_world_state", "params": {}},
                "actor": "assistant",
            },
        )
        self.assertEqual(r.status_code, 200)
        body = r.json()
        self.assertTrue(body["ok"])
        self.assertIn("npcs", body["data"])

    def test_system_create_item_and_assistant_denied(self) -> None:
        client = _client()
        payload = {
            "command": {
                "command_type": "create_item",
                "params": {
                    "item_id": "test_item",
                    "name": "test",
                    "icon_path": "res://graphics/icons/apple.png",
                },
            },
            "actor": "assistant",
        }
        denied = client.post("/v1/game/command", json=payload)
        self.assertEqual(denied.status_code, 200)
        self.assertEqual(denied.json()["code"], "denied")

        payload["actor"] = "system"
        ok = client.post("/v1/game/command", json=payload)
        self.assertTrue(ok.json()["ok"])

    def test_game_command_send_message(self) -> None:
        r = _client().post(
            "/v1/game/command",
            json={
                "command": {
                    "command_type": "send_message",
                    "params": {"player_id": "player_001", "text": "hello"},
                },
                "actor": "assistant",
            },
        )
        self.assertTrue(r.json()["ok"])

    def test_context(self) -> None:
        r = _client().get("/v1/context?player_id=player_001&location=farm")
        self.assertEqual(r.status_code, 200)
        data = r.json()["data"]
        self.assertIn("world", data)
        self.assertIn("inventory", data)
        self.assertIn("memories", data)


if __name__ == "__main__":
    unittest.main()
