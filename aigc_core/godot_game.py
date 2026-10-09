"""通向正在运行 Godot 的真实游戏适配器（文件 inbox 命令通道）。

主控侧（Python）把 AI 的命令序列化成 JSONL，追加写到 Godot 能读的
`user://commands/inbox.jsonl`；Godot 侧每帧读取并执行后清空该文件。

和 Telemetry 的单向“Godot -> Python 日志文件”对称：这一层是
“Python 主控 -> Godot 命令文件”的入站通道。后续可平滑替换为
WebSocket/TCP，接口（GameAdapter）不变。
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any, Dict, Optional

from .game import GameAdapter
from .models import ActionResult, GameCommand, GameQuery

# Godot user:// 实际落盘位置：%APPDATA%/Godot/app_userdata/StardewValley
# 2026-09-10 修:从无用户环境变量的上下文拉起时 KeyError:'APPDATA' 崩溃(见 ctx8000.loge),加回退
_APPDATA = os.environ.get("APPDATA") or str(Path.home() / "AppData" / "Roaming")
DEFAULT_APP_USERDATA = (
    Path(_APPDATA) / "Godot" / "app_userdata" / "StardewValley"
)
DEFAULT_INBOX_PATH = DEFAULT_APP_USERDATA / "commands" / "inbox.jsonl"


class FileCommandGameAdapter(GameAdapter):
    """把主控命令追加写入 Godot inbox 文件的适配器。

    命令是否真的被执行，需要等 Godot 读取；这里只负责“排队成功”。
    query 目前无法在文件通道实现，返回未实现。
    """

    def __init__(
        self,
        inbox_path: Optional[os.PathLike[str] | str] = None,
        actor: str = "system",
    ) -> None:
        self.inbox_path = (
            Path(inbox_path).expanduser() if inbox_path else DEFAULT_INBOX_PATH
        )
        self.actor = actor

    def query(self, query: GameQuery) -> ActionResult:
        # 文件通道是单向命令队列；真实只读查询建议后续走 WebSocket/快照文件
        return ActionResult(
            ok=False,
            code="not_implemented",
            message=f"file command channel cannot answer query: {query.query_type}",
        )

    def command(self, command: GameCommand) -> ActionResult:
        payload: Dict[str, Any] = {
            "command_id": command.command_id,
            "command_type": command.command_type,
            "params": command.params,
            "actor": self.actor,
            "queued_at": time.time(),
        }
        try:
            self.inbox_path.parent.mkdir(parents=True, exist_ok=True)
            with self.inbox_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(payload, ensure_ascii=False) + "\n")
        except OSError as exc:  # pragma: no cover - 依赖本机路径
            return ActionResult(
                ok=False, code="error", message=f"cannot write inbox: {exc}"
            )
        return ActionResult(
            ok=True,
            code="queued",
            message="command queued to godot inbox",
            data={"inbox_path": str(self.inbox_path), "queued": payload},
        )
