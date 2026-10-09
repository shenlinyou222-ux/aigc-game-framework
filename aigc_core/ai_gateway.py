"""AI 上层接口占位。

本文件只定义 AI 层如何反向依赖主控，不实现具体模型。
后续接入 Ollama/向量库/意图编排时，实现 AILayer 即可。
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, List

from .controller import MainController
from .models import ActionResult, ActorRole, GameCommand, GameQuery


class AILayer(ABC):
    """所有 AI 能力都通过 MainController 操作游戏和记忆。"""

    def __init__(self, controller: MainController) -> None:
        self.controller = controller

    @abstractmethod
    async def handle_player_input(self, player_id: str, text: str) -> Dict[str, Any]:
        """处理玩家输入，返回结构化响应/动作列表。"""

    # 下面两个方法演示 AI 层唯一可用的游戏接口：
    # 只读用 controller.game_query
    # 写操作用 controller.game_command
    def _read_game(self, query_type: str, params: Dict[str, Any]) -> ActionResult:
        return self.controller.game_query(
            GameQuery(query_type=query_type, params=params),
            actor=ActorRole.ASSISTANT,
        )

    def _write_game(self, command_type: str, params: Dict[str, Any]) -> ActionResult:
        return self.controller.game_command(
            GameCommand(command_type=command_type, params=params),
            actor=ActorRole.ASSISTANT,
        )
