"""本地演示服务：把 aigc_core 的主控能力用 HTTP 暴露出来。

运行：
    .venv\\Scripts\\python.exe -m uvicorn demo_server:app --port 8765

打开 http://127.0.0.1:8765/ 是说明页；
http://127.0.0.1:8765/docs 是 FastAPI 自动生成的接口文档；
http://127.0.0.1:8765/demo/run 会跑一遍"AI 读世界 → 下命令 → 越权被拒"的完整链路。

注意：这是演示壳，不是生产入口。真实入口见 aigc_core/api.py 的 create_app。
"""

from __future__ import annotations

from fastapi import FastAPI
from fastapi.responses import HTMLResponse

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
from aigc_core.api import create_app

# 用内存态主控 + 模拟游戏，演示不落盘、不依赖模型
game = MockGame()
controller = MainController(game_adapter=game)
app: FastAPI = create_app(controller=controller)


@app.get("/", response_class=HTMLResponse)
def index() -> str:
    return """<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<title>AIGC Game Framework · 本地演示</title>
<style>
  body { font-family: -apple-system, "Segoe UI", "Microsoft YaHei", sans-serif;
         max-width: 780px; margin: 48px auto; padding: 0 20px; color: #1a1a1a; line-height: 1.7; }
  h1 { font-size: 24px; margin-bottom: 4px; }
  .sub { color: #666; margin-top: 0; }
  code { background: #f3f4f6; padding: 1px 5px; border-radius: 4px; font-size: 13px; }
  .card { border: 1px solid #e5e7eb; border-radius: 10px; padding: 16px 20px; margin: 18px 0; }
  a { color: #0b62d0; }
  .tag { display: inline-block; font-size: 12px; background: #eef2ff; color: #3730a3;
         border-radius: 999px; padding: 1px 9px; margin-right: 6px; }
</style>
</head>
<body>
  <h1>AIGC Game Framework · 本地演示</h1>
  <p class="sub">AI 不侵入游戏主循环，只能通过主控的受限接口读记忆、下指令。</p>

  <div class="card">
    <span class="tag">一键</span>
    <a href="/demo/run">/demo/run</a>
    <p>跑一遍完整链路：写入玩家事件 → AI 读世界状态 → 整理背包（允许）→ 刷物品（拒绝）。</p>
  </div>

  <div class="card">
    <span class="tag">接口</span>
    <a href="/docs">/docs</a>
    <p>FastAPI 自动文档，可展开全部 <code>/v1/*</code> 与 <code>/ctx/*</code> 接口。</p>
  </div>

  <div class="card">
    <span class="tag">健康</span>
    <a href="/health">/health</a>
    <p>返回 <code>{"status":"ok"}</code>。</p>
  </div>

  <div class="card">
    <span class="tag">自测</span>
    <p><code>.venv\\Scripts\\python.exe -m pytest tests -q</code> → 223 passed</p>
  </div>
</body>
</html>"""


@app.get("/demo/run")
def demo_run() -> dict:
    """跑一遍 run_demo.py 的链路，返回结构化结果。"""
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

    world = controller.game_query(
        GameQuery(query_type="get_world_state", params={"player_id": "player_001"}),
        actor=ActorRole.ASSISTANT,
    )
    organize = controller.game_command(
        GameCommand(command_type="organize_inventory", params={"player_id": "player_001"}),
        actor=ActorRole.ASSISTANT,
    )
    cheat = controller.game_command(
        GameCommand(
            command_type="give_item",
            params={"player_id": "player_001",
                    "item": {"id": "cheat_sword", "name": "作弊之剑"}},
        ),
        actor=ActorRole.ASSISTANT,
    )

    return {
        "steps": [
            {"step": "AI 读世界状态", "ok": world.ok},
            {"step": "AI 整理背包", "ok": organize.ok},
            {"step": "AI 刷物品（应被拒）", "ok": cheat.ok, "code": getattr(cheat, "code", None)},
        ],
        "note": "第三条 ok=false 是预期：主控只放行白名单命令。",
    }
