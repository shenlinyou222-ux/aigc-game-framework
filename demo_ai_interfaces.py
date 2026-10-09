"""AI 实测：接口 -> 特殊向量库召回 -> 真实调用 HTTP 接口。

演示闭环：
1. 把可调用接口(schema+描述)向量化进“特殊接口向量库”(InterfaceVectorStore)；
2. 作为 AI 接收自然语言意图 -> 从该向量库召回最相关接口；
3. 通过 FastAPI(TestClient=真实 HTTP 层) 调用被召回的接口；
4. 每次调用再作为一条日志向量写入日志向量库(LogVectorStore)；
5. 打印召回/调用结果，方便肉眼核对。

默认用确定性 HashEmbedder(无需外部服务)；接 Ollama 时换成
OllamaLogEmbedder(model="bge-m3") 即可。
"""

from __future__ import annotations

import json
from pathlib import Path

from fastapi.testclient import TestClient

from aigc_core import (
    HashEmbedder,
    InterfaceVectorStore,
    LogVectorStore,
    default_interface_specs,
)
from aigc_core.api import create_app
from aigc_core.models import GameLogRecord

BASE = Path(__file__).resolve().parent
INTERFACES_DB = BASE / "data_interfaces.jsonl"
CALLS_DB = BASE / "data_interface_calls.jsonl"


def _act_as_ai(text: str, store: InterfaceVectorStore) -> str:
    print(f"\n[AI] 意图: {text}")
    hits = store.query(text, top_k=3)
    for h in hits:
        print(f"     召回 {h.score:.3f}  {h.spec.name}  <- {h.spec.description[:36]}")
    return hits[0].spec.name if hits else ""


def main() -> None:
    # ---- 1) 接口特殊向量库：把接口 schema 向量化 ----
    print("=" * 72)
    print("1) 建立“接口即向量”特殊向量库")
    print("=" * 72)
    emb = HashEmbedder(dim=128)
    iface = InterfaceVectorStore(path=INTERFACES_DB, embedder=emb)
    iface.add_many(default_interface_specs())
    print(f"   入库接口数 = {iface.count()} -> {INTERFACES_DB.name}")
    print("   接口:", ", ".join(iface.names()))

    # ---- 2) 真实 HTTP 层（FastAPI TestClient = AI/客户端同入口）----
    print("\n" + "=" * 72)
    print("2) 作为 AI 召回接口并真实调用 HTTP 接口")
    print("=" * 72)
    client = TestClient(create_app())
    call_logs = LogVectorStore(path=CALLS_DB)

    def call(interface: str, params: dict, actor: str) -> dict:
        # 每条被召回的接口真实打出去（写命令在演示里由主控以 system 身份落）
        r = client.post(
            "/v1/game/command",
            json={"command": {"command_type": interface, "params": params}, "actor": actor},
        )
        body = r.json()
        print(f"   -> POST /v1/game/command {interface} actor={actor}")
        print(f"      resp ok={body.get('ok')} code={body.get('code')} data={body.get('data')}")
        # 每次调用作为一条向量日志进入日志向量库
        call_logs.add(
            GameLogRecord(
                event_id=f"interface.called.{interface}.{len(call_logs.all())}",
                event_type="interface.called",
                time="",
                unix_time=0.0,
                data={
                    "interface": interface,
                    "actor": actor,
                    "params": params,
                    "result": body,
                },
                semantic_vector=emb(json.dumps({"interface": interface, "params": params}, ensure_ascii=False)),
            )
        )
        return body

    # ---- 3) 跑几条真实意图 ----
    cases = [
        ("创建一种叫 test 的新物品种类", "create_item", {"item_id": "test_item", "name": "test", "icon_path": "res://graphics/icons/apple.png", "count": 5}, "system"),
        ("把刚才的 test 物品发给玩家", "give_item", {"player_id": "player_001", "item_id": "test_item", "count": 1}, "system"),
        ("给玩家发条消息", "send_message", {"player_id": "player_001", "text": "这是 AI 实测。"}, "assistant"),
        ("换一首轻松的背景音乐", "play_music", {"track": "rain", "volume_db": -18}, "assistant"),
    ]
    for text, expect, params, actor in cases:
        picked = _act_as_ai(text, iface)
        marker = "✓" if picked == expect else f"✗(期望 {expect})"
        print(f"   召回选定: {picked}  {marker}")
        call(picked, params, actor)

    # ---- 4) 汇总：接口向量库 & 调用日志向量库 ----
    print("\n" + "=" * 72)
    print("3) 收尾：接口向量库 / 调用日志向量库 状态")
    print("=" * 72)
    print(f"   接口向量库 条数 = {iface.count()}")
    print(f"   调用日志向量 条数 = {call_logs.count()}  -> {CALLS_DB.name}")
    for r in call_logs.all():
        print("   -", r.event_id, r.data["interface"], r.data.get("result", {}).get("code"))


if __name__ == "__main__":
    main()
