# -*- coding: utf-8 -*-
"""AI->游戏调用日志测试：ai.command 无论成败必记；ai.query 仅失败记。"""

from __future__ import annotations

from aigc_core import MockGame
from aigc_core.controller import MainController
from aigc_core.log_vector_store import LogVectorStore
from aigc_core.models import ActorRole, GameCommand, GameQuery


def _ctrl() -> MainController:
    return MainController(game_adapter=MockGame(), log_store=LogVectorStore())


def test_successful_command_is_logged():
    c = _ctrl()
    r = c.game_command(
        GameCommand(command_type="send_message", params={"player_id": "p1", "text": "你好"}),
        actor=ActorRole.ASSISTANT,
    )
    assert r.ok
    hits = c.query_logs(LogVectorQueryLite(tags=["src:ai", "cmd:send_message"]))
    assert len(hits) == 1
    rec = hits[0].record
    assert rec.event_type == "ai.command"
    assert rec.event_id.startswith("ai.command.")
    assert rec.data["result"]["ok"] is True
    assert "ok" in rec.tags and "src:ai" in rec.tags
    assert rec.unix_time > 0  # 时间戳必须真实（老 interface.called 的教训）


def test_denied_command_is_logged_with_error_tag():
    c = _ctrl()
    r = c.game_command(
        GameCommand(command_type="give_item", params={}),  # assistant 无权 give_item
        actor=ActorRole.ASSISTANT,
    )
    assert not r.ok and r.code == "denied"
    hits = c.query_logs(LogVectorQueryLite(tags=["err:denied"]))
    assert len(hits) == 1
    rec = hits[0].record
    assert rec.data["result"]["code"] == "denied"
    assert f"err:denied" in rec.tags


def test_failed_query_logged_success_query_not():
    c = _ctrl()
    # 成功查询：不落库
    ok = c.game_query(GameQuery(query_type="get_world_state", params={}), actor=ActorRole.ASSISTANT)
    assert ok.ok
    # 被拒查询：落库
    bad = c.game_query(GameQuery(query_type="create_item", params={}), actor=ActorRole.ASSISTANT)
    assert not bad.ok
    all_ai = c.query_logs(LogVectorQueryLite(tags=["src:ai"]))
    queries = [h.record for h in all_ai if h.record.event_type == "ai.query"]
    assert len(queries) == 1
    assert queries[0].data["result"]["code"] == "denied"
    commands = [h.record for h in all_ai if h.record.event_type == "ai.command"]
    assert len(commands) == 0


def test_command_id_join():
    """event_id 含 command_id，可与 Godot 侧 command.executed 因果关联。"""
    from aigc_core.models import LogVectorQuery

    c = _ctrl()
    cmd = GameCommand(command_type="send_message", params={"player_id": "p1", "text": "hi"})
    c.game_command(cmd, actor=ActorRole.ASSISTANT)
    rec = c.get_log(f"ai.command.{cmd.command_id}")
    assert rec is not None and rec.data["command"] == "send_message"


from aigc_core.models import LogVectorQuery as LogVectorQueryLite  # noqa: E402
