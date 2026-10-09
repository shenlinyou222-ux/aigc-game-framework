"""游戏适配层。

真实游戏接入时实现 GameAdapter 即可：
- query() 对应只读接口
- command() 对应写操作接口

当前提供一个 MockGame 用于在没有真实游戏时验证框架。
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

from .models import ActionResult, GameCommand, GameQuery


class GameAdapter(ABC):
    """真实游戏需要实现的接口。"""

    @abstractmethod
    def query(self, query: GameQuery) -> ActionResult:
        """处理只读查询。"""

    @abstractmethod
    def command(self, command: GameCommand) -> ActionResult:
        """处理写操作。"""


class MockGame(GameAdapter):
    """模拟一个 2D 农场/世界游戏。

    数据尽量贴近“星露谷式 Demo”：玩家、背包、NPC、设置、邮件。
    """

    def __init__(self) -> None:
        self.players: Dict[str, Dict[str, Any]] = {
            "player_001": {
                "name": "测试玩家",
                "position": [12, 34],
                "health": 100,
                "energy": 80,
                "location": "farm",
            }
        }
        self.inventory: Dict[str, List[Dict[str, Any]]] = {
            "player_001": [
                {"id": "item_001", "name": "铜矿石", "count": 3},
                {"id": "item_002", "name": "芜菁种子", "count": 5},
                {"id": "item_003", "name": "铁矿", "count": 2},
            ]
        }
        self.npcs: Dict[str, Dict[str, Any]] = {
            "npc_777": {
                "name": "NPC-777",
                "position": [14, 34],
                "mood": "neutral",
                "last_dialogue": "",
                "confidence": 0.8,
                "faction": "village",
            },
            "npc_778": {
                "name": "NPC-778",
                "position": [10, 36],
                "mood": "happy",
                "last_dialogue": "",
                "confidence": 0.9,
                "faction": "village",
            },
        }
        self.settings: Dict[str, Any] = {"volume": 0.7, "language": "zh", "show_emote": True}
        self.mailbox: Dict[str, List[Dict[str, Any]]] = {}
        self.command_log: List[str] = []
        # 动态切歌：模拟 Godot BGM 状态
        self.music: Dict[str, Any] = {
            "track": "day",
            "playing": True,
            "volume_db": -20.0,
        }
        # 动态物品注册表：AI 创建的新物品种类（id -> {id,name,icon_path,count}）
        self.custom_items: Dict[str, Dict[str, Any]] = {}

    def _ensure_mailbox(self, player_id: str) -> None:
        self.mailbox.setdefault(player_id, [])

    def query(self, query: GameQuery) -> ActionResult:
        qtype = query.query_type
        params = query.params
        if qtype == "get_world_state":
            return ActionResult(data={
                "location": "farm",
                "time": 600,
                "players": list(self.players.values()),
                "npcs": list(self.npcs.values()),
            })
        if qtype == "get_player_state":
            player_id = params.get("player_id", "player_001")
            player = self.players.get(player_id)
            if not player:
                return ActionResult(ok=False, code="not_found", message=f"player not found: {player_id}")
            return ActionResult(data={"player": player, "inventory": self.inventory.get(player_id, [])})
        if qtype == "get_inventory":
            player_id = params.get("player_id", "player_001")
            items = self.inventory.get(player_id, [])
            return ActionResult(data={"player_id": player_id, "items": items})
        if qtype == "get_npc_state":
            npc_id = params.get("npc_id", "npc_777")
            npc = self.npcs.get(npc_id)
            if not npc:
                return ActionResult(ok=False, code="not_found", message=f"npc not found: {npc_id}")
            return ActionResult(data={"npc": npc})
        if qtype == "get_game_setting":
            key = params.get("key")
            if key:
                return ActionResult(data={"key": key, "value": self.settings.get(key)})
            return ActionResult(data={"settings": self.settings})
        if qtype == "list_entities":
            return ActionResult(data={
                "players": list(self.players.keys()),
                "npcs": list(self.npcs.keys()),
            })
        return ActionResult(ok=False, code="not_found", message=f"unknown query: {qtype}")

    def command(self, command: GameCommand) -> ActionResult:
        ctype = command.command_type
        params = command.params
        self.command_log.append(command.model_dump_json())

        if ctype == "send_message":
            player_id = params.get("player_id", "player_001")
            text = params.get("text", "")
            self._ensure_mailbox(player_id)
            self.mailbox[player_id].append({"sender": params.get("sender", "system"), "text": text})
            return ActionResult(data={"player_id": player_id, "text": text})

        if ctype == "send_npc_dialogue":
            npc_id = params.get("npc_id", "npc_777")
            text = params.get("text", "")
            npc = self.npcs.get(npc_id)
            if not npc:
                return ActionResult(ok=False, code="not_found", message=f"npc not found: {npc_id}")
            npc["last_dialogue"] = text
            return ActionResult(data={"npc_id": npc_id, "text": text})

        if ctype == "show_emote":
            npc_id = params.get("npc_id", "npc_777")
            emote = params.get("emote", "neutral")
            if npc_id in self.npcs:
                self.npcs[npc_id]["mood"] = emote
            return ActionResult(data={"npc_id": npc_id, "emote": emote})

        if ctype == "organize_inventory":
            player_id = params.get("player_id", "player_001")
            items = self.inventory.get(player_id, [])
            items.sort(key=lambda x: (x.get("name", ""), x.get("id", "")))
            return ActionResult(data={"player_id": player_id, "items": items})

        if ctype == "set_game_setting":
            key = params.get("key")
            value = params.get("value")
            if key is None:
                return ActionResult(ok=False, code="invalid", message="missing key")
            self.settings[key] = value
            return ActionResult(data={"key": key, "value": value})

        if ctype == "create_item":
            item_id = params.get("item_id")
            name = params.get("name", item_id)
            icon_path = params.get("icon_path", "")
            count = int(params.get("count", 0))
            if not item_id:
                return ActionResult(ok=False, code="invalid", message="missing item_id")
            self.custom_items[item_id] = {
                "id": item_id,
                "name": name,
                "icon_path": icon_path,
                "count": count,
            }
            return ActionResult(
                data={"item_id": item_id, "item": self.custom_items[item_id]}
            )

        if ctype == "play_music":
            track = params.get("track")
            if not track:
                return ActionResult(ok=False, code="invalid", message="missing track")
            self.music["track"] = track
            if "volume_db" in params:
                self.music["volume_db"] = float(params["volume_db"])
            if "playing" in params:
                self.music["playing"] = bool(params["playing"])
            return ActionResult(data={"track": self.music["track"]})

        if ctype == "give_item":
            player_id = params.get("player_id", "player_001")
            item = params.get("item")
            item_id = params.get("item_id")
            count = int(params.get("count", 1))
            if isinstance(item, dict):
                # 兼容旧格式：直接给一条物品记录
                self.inventory.setdefault(player_id, []).append(item)
                return ActionResult(data={"player_id": player_id, "item": item})
            if item_id:
                # 动态物品：从注册表取定义，加入玩家背包
                custom = self.custom_items.get(item_id)
                if custom is None:
                    return ActionResult(
                        ok=False, code="not_found",
                        message=f"unknown custom item: {item_id}",
                    )
                given = {"id": item_id, "name": custom["name"], "count": count}
                self.inventory.setdefault(player_id, []).append(given)
                return ActionResult(data={"player_id": player_id, "item": given})
            return ActionResult(ok=False, code="invalid", message="missing item/item_id")

        if ctype == "set_npc_confidence":
            npc_id = params.get("npc_id", "npc_777")
            confidence = params.get("confidence")
            npc = self.npcs.get(npc_id)
            if not npc or confidence is None:
                return ActionResult(ok=False, code="invalid", message="invalid npc/confidence")
            npc["confidence"] = float(confidence)
            return ActionResult(data={"npc_id": npc_id, "confidence": npc["confidence"]})

        if ctype == "save_game":
            # Mock 无世界可存; 真实链路走 FileCommandGameAdapter -> Godot SaveGame
            return ActionResult(data={
                "saved": True, "mock": True,
                "slot": params.get("slot", "auto"),
            })

        if ctype == "load_game":
            # 权限在 controller.game_command 拦截(load_game 仅 system); Mock 只回执
            return ActionResult(data={"loaded": True, "mock": True})

        if ctype == "advance_mainline":
            return ActionResult(ok=False, code="denied", message="advance_mainline not implemented in mock")

        return ActionResult(ok=False, code="not_found", message=f"unknown command: {ctype}")
