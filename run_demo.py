"""演示：没有真实 AI，只用规则模拟“AI 上层通过主控操作游戏”。

运行：
    E:\\py312\\python.exe run_demo.py
"""

from __future__ import annotations

from aigc_core import (
    ActorRole,
    GameCommand,
    GameEvent,
    GameQuery,
    Lifecycle,
    MainController,
    MockGame,
    SourceType,
    WriterRole,
)

# 1. 真实游戏适配器
game = MockGame()
controller = MainController(game_adapter=game)

# 2. 游戏/感知组件把事件写入主控
controller.record_event(
    GameEvent(
        event_type="player_chat",
        source_type=SourceType.EYEWITNESS,
        source_faction="player",
        location="farm",
        entity_id="player_001",
        lifecycle=Lifecycle.B,
        confidence=0.95,
        payload={"text": "帮我把背包整理一下"},
    ),
    writer=WriterRole.SENSOR,
)

# 3. 上层 AI 先读取上下文
world = controller.game_query(
    GameQuery(query_type="get_world_state", params={"player_id": "player_001"}),
    actor=ActorRole.ASSISTANT,
)
print("[AI] 世界状态 ok =", world.ok)

# 4. AI 发出有限操作命令
organize = controller.game_command(
    GameCommand(
        command_type="organize_inventory",
        params={"player_id": "player_001"},
    ),
    actor=ActorRole.ASSISTANT,
)
print("[AI] 整理背包 ok =", organize.ok)

# 5. AI 如果试图刷物品，会被主控拒绝
give = controller.game_command(
    GameCommand(
        command_type="give_item",
        params={"player_id": "player_001", "item": {"id": "cheat_sword", "name": "作弊之剑"}},
    ),
    actor=ActorRole.ASSISTANT,
)
print("[AI] 刷物品 ok =", give.ok, "code =", give.code)

# 6. 查看主控自动补的系统命令事件
from aigc_core import MemoryQuery  # noqa: E402

events = controller.query_memory(MemoryQuery(event_type="system.command_executed"))
print("[主控] 系统命令事件数 =", len(events))
for e in events:
    print("   -", e.payload.get("command"), "by", e.payload.get("actor"))
