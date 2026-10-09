"""框架冒烟测试：事件向量格式、主控权限、游戏操作接口。"""

from __future__ import annotations

import unittest

from aigc_core import (
    MainController,
    MockGame,
    ActorRole,
    GameCommand,
    GameEvent,
    GameQuery,
    Lifecycle,
    MemoryQuery,
    SourceType,
    WriterRole,
)
from aigc_core.memory import MemoryStoreError


class EventVectorFormatTests(unittest.TestCase):
    def test_game_event_to_memory_entry(self) -> None:
        event = GameEvent(
            event_type="player_chat",
            source_type=SourceType.EYEWITNESS,
            source_faction="player",
            location="village",
            entity_id="player_001",
            lifecycle=Lifecycle.B,
            confidence=0.95,
            payload={"text": "你好，小灰"},
        )
        entry = event.to_memory_entry()
        self.assertEqual(entry.exact_fields.type, "player_chat")
        self.assertEqual(entry.exact_fields.entity_id, "player_001")
        self.assertEqual(entry.lifecycle, Lifecycle.B)
        self.assertEqual(entry.payload["text"], "你好，小灰")

    def test_s_lifecycle_rejects_model_writer(self) -> None:
        ctrl = MainController()
        with self.assertRaises(MemoryStoreError):
            ctrl.record_event(
                GameEvent(
                    event_type="mainline_done",
                    source_type=SourceType.SYSTEM,
                    lifecycle=Lifecycle.S,
                    payload={"flag": True},
                ),
                writer=WriterRole.MODEL,
            )

    def test_s_lifecycle_accepts_program_writer(self) -> None:
        ctrl = MainController()
        entry = ctrl.record_event(
            GameEvent(
                event_type="mainline_done",
                source_type=SourceType.SYSTEM,
                lifecycle=Lifecycle.S,
                payload={"flag": True},
            ),
            writer=WriterRole.PROGRAM,
        )
        self.assertEqual(ctrl.get_memory(entry.event_id).lifecycle, Lifecycle.S)

    def test_memory_query_filter(self) -> None:
        ctrl = MainController()
        ctrl.record_event(
            GameEvent(event_type="player_chat", location="farm", entity_id="player_001", payload={"text": "a"}),
            writer=WriterRole.PROGRAM,
        )
        ctrl.record_event(
            GameEvent(event_type="npc_interact", location="village", entity_id="npc_777", payload={"text": "b"}),
            writer=WriterRole.PROGRAM,
        )
        result = ctrl.query_memory(MemoryQuery(event_type="player_chat"))
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0].exact_fields.type, "player_chat")


class PermissionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.controller = MainController(game_adapter=MockGame())

    def test_assistant_can_send_message(self) -> None:
        result = self.controller.game_command(
            GameCommand(
                command_type="send_message",
                params={"player_id": "player_001", "text": "你好，我是小灰"},
            ),
            actor=ActorRole.ASSISTANT,
        )
        self.assertTrue(result.ok)
        game = self.controller.game
        self.assertEqual(game.mailbox["player_001"][-1]["text"], "你好，我是小灰")

    def test_assistant_can_play_music(self) -> None:
        result = self.controller.game_command(
            GameCommand(command_type="play_music", params={"track": "rain"}),
            actor=ActorRole.ASSISTANT,
        )
        self.assertTrue(result.ok)
        self.assertEqual(self.controller.game.music["track"], "rain")

    def test_assistant_cannot_create_item(self) -> None:
        result = self.controller.game_command(
            GameCommand(
                command_type="create_item",
                params={"item_id": "test_item", "name": "test", "icon_path": "res://x.png"},
            ),
            actor=ActorRole.ASSISTANT,
        )
        self.assertFalse(result.ok)
        self.assertEqual(result.code, "denied")

    def test_system_can_create_item(self) -> None:
        result = self.controller.game_command(
            GameCommand(
                command_type="create_item",
                params={"item_id": "test_item", "name": "test", "icon_path": "res://graphics/icons/apple.png"},
            ),
            actor=ActorRole.SYSTEM,
        )
        self.assertTrue(result.ok)
        self.assertIn("test_item", self.controller.game.custom_items)

    def test_give_custom_item_requires_created(self) -> None:
        result = self.controller.game_command(
            GameCommand(
                command_type="give_item",
                params={"player_id": "player_001", "item_id": "not_created", "count": 1},
            ),
            actor=ActorRole.SYSTEM,
        )
        self.assertFalse(result.ok)
        self.assertEqual(result.code, "not_found")

    def test_give_created_custom_item(self) -> None:
        self.controller.game_command(
            GameCommand(
                command_type="create_item",
                params={"item_id": "test_item", "name": "test", "icon_path": "res://x.png"},
            ),
            actor=ActorRole.SYSTEM,
        )
        result = self.controller.game_command(
            GameCommand(
                command_type="give_item",
                params={"player_id": "player_001", "item_id": "test_item", "count": 3},
            ),
            actor=ActorRole.SYSTEM,
        )
        self.assertTrue(result.ok)
        last = self.controller.game.inventory["player_001"][-1]
        self.assertEqual(last["id"], "test_item")
        self.assertEqual(last["name"], "test")
        self.assertEqual(last["count"], 3)

    def test_assistant_cannot_give_item(self) -> None:
        result = self.controller.game_command(
            GameCommand(
                command_type="give_item",
                params={"player_id": "player_001", "item": {"id": "legend_sword", "name": "传说之剑"}},
            ),
            actor=ActorRole.ASSISTANT,
        )
        self.assertFalse(result.ok)
        self.assertEqual(result.code, "denied")

    def test_system_can_give_item(self) -> None:
        result = self.controller.game_command(
            GameCommand(
                command_type="give_item",
                params={"player_id": "player_001", "item": {"id": "legend_sword", "name": "传说之剑"}},
            ),
            actor=ActorRole.SYSTEM,
        )
        self.assertTrue(result.ok)

    def test_assistant_cannot_delete_memory(self) -> None:
        with self.assertRaises(PermissionError):
            self.controller.delete_memory("anything", actor=ActorRole.ASSISTANT)

    def test_game_query_read_inventory(self) -> None:
        result = self.controller.game_query(
            GameQuery(query_type="get_inventory", params={"player_id": "player_001"}),
            actor=ActorRole.ASSISTANT,
        )
        self.assertTrue(result.ok)
        self.assertEqual(len(result.data["items"]), 3)


class LifecyclePermissionExtraTests(unittest.TestCase):
    def test_a_lifecycle_rejects_model_writer(self) -> None:
        ctrl = MainController()
        with self.assertRaises(MemoryStoreError):
            ctrl.record_event(
                GameEvent(event_type="npc_interact", lifecycle=Lifecycle.A),
                writer=WriterRole.MODEL,
            )

    def test_system_can_delete_memory(self) -> None:
        ctrl = MainController()
        entry = ctrl.record_event(
            GameEvent(event_type="player_chat", payload={"text": "x"}),
            writer=WriterRole.PROGRAM,
        )
        self.assertTrue(ctrl.delete_memory(entry.event_id, actor=ActorRole.SYSTEM))


class MockGameQueryCommandTests(unittest.TestCase):
    def setUp(self) -> None:
        self.game = MockGame()
        self.ctrl = MainController(game_adapter=self.game)

    def test_query_player_state(self) -> None:
        r = self.ctrl.game_query(GameQuery(query_type="get_player_state", params={"player_id": "player_001"}))
        self.assertTrue(r.ok)
        self.assertEqual(r.data["player"]["name"], "测试玩家")

    def test_query_player_state_missing(self) -> None:
        r = self.ctrl.game_query(GameQuery(query_type="get_player_state", params={"player_id": "nobody"}))
        self.assertFalse(r.ok)
        self.assertEqual(r.code, "not_found")

    def test_query_npc_state(self) -> None:
        r = self.ctrl.game_query(GameQuery(query_type="get_npc_state", params={"npc_id": "npc_777"}))
        self.assertTrue(r.ok)
        self.assertEqual(r.data["npc"]["faction"], "village")

    def test_query_game_setting_single_and_all(self) -> None:
        r = self.ctrl.game_query(GameQuery(query_type="get_game_setting", params={"key": "volume"}))
        self.assertEqual(r.data["value"], 0.7)
        r2 = self.ctrl.game_query(GameQuery(query_type="get_game_setting", params={}))
        self.assertIn("settings", r2.data)

    def test_query_list_entities(self) -> None:
        r = self.ctrl.game_query(GameQuery(query_type="list_entities", params={}))
        self.assertIn("npc_777", r.data["npcs"])

    def test_unknown_query(self) -> None:
        r = self.ctrl.game_query(GameQuery(query_type="what", params={}))
        self.assertFalse(r.ok)
        self.assertEqual(r.code, "not_found")

    def test_command_set_game_setting(self) -> None:
        r = self.ctrl.game_command(GameCommand(command_type="set_game_setting", params={"key": "volume", "value": 0.3}), actor=ActorRole.SYSTEM)
        self.assertTrue(r.ok)
        self.assertEqual(self.game.settings["volume"], 0.3)

    def test_command_show_emote(self) -> None:
        r = self.ctrl.game_command(GameCommand(command_type="show_emote", params={"npc_id": "npc_777", "emote": "happy"}), actor=ActorRole.SYSTEM)
        self.assertTrue(r.ok)
        self.assertEqual(self.game.npcs["npc_777"]["mood"], "happy")

    def test_command_advance_mainline_denied(self) -> None:
        r = self.ctrl.game_command(GameCommand(command_type="advance_mainline", params={}), actor=ActorRole.SYSTEM)
        self.assertFalse(r.ok)
        self.assertEqual(r.code, "denied")

    def test_command_unknown(self) -> None:
        r = self.ctrl.game_command(GameCommand(command_type="does_not_exist", params={}), actor=ActorRole.SYSTEM)
        self.assertFalse(r.ok)
        self.assertEqual(r.code, "not_found")


if __name__ == "__main__":
    unittest.main()
