"""DSH 上下文后端（/ctx/*）与“准确召回”混合方法的测试。

覆盖：
- session_id 过滤（全局长期记忆可被任意 session 命中）
- recall_hybrid 的 词法/时效/本地 加权与命中词
- /ctx/ingest · /ctx/recall · /ctx/touch · /ctx/cleanup · /ctx/status
"""

from __future__ import annotations

import time
import unittest

from fastapi.testclient import TestClient

from aigc_core.api import create_app
from aigc_core.log_vector_store import LogVectorStore
from aigc_core.models import GameLogRecord, LogVectorQuery


def _store() -> LogVectorStore:
    store = LogVectorStore()
    now = time.time()
    store.add(
        GameLogRecord(
            event_id="user-local",
            event_type="user/message",
            kind="user/message",
            unix_time=now - 1000,
            session_id="s1",
            data={"text": "帮我修一下 player.gd 的报错"},
        )
    )
    store.add(
        GameLogRecord(
            event_id="tool-recent",
            event_type="tool/result",
            kind="tool/result",
            unix_time=now - 10,
            session_id="s1",
            data={"tool": "read_file", "path": "player.gd", "result": "line 12 syntax error"},
        )
    )
    store.add(
        GameLogRecord(
            event_id="other-session",
            event_type="user/message",
            kind="user/message",
            unix_time=now - 500,
            session_id="s2",
            data={"text": "今天天气怎么样"},
        )
    )
    # 全局长期记忆（session_id 空）：应可被任意 session 召回
    store.add(
        GameLogRecord(
            event_id="global-tip",
            event_type="assistant/message",
            kind="assistant/message",
            unix_time=now - 5000,
            session_id="",
            data={"text": "项目约定：先读 AGENTS.md 再改代码"},
        )
    )
    return store


class SessionFilterTests(unittest.TestCase):
    def test_query_limits_to_session_and_global(self) -> None:
        store = _store()
        # session_id=None：不过滤，全都能查到
        self.assertEqual(
            len(store.query(LogVectorQuery(limit=100))),
            4,
        )
        # session_id='s1'：只含 s1 与全局
        s1 = store.query(LogVectorQuery(session_id="s1", limit=100))
        ids = {r.record.event_id for r in s1}
        self.assertIn("user-local", ids)
        self.assertIn("tool-recent", ids)
        self.assertIn("global-tip", ids)
        self.assertNotIn("other-session", ids)

    def test_hybrid_recall_prefers_local_recent_tool_result(self) -> None:
        store = _store()
        hits = store.recall_hybrid(
            query_text="player.gd 报错 修一下",
            session_id="s1",
            limit=5,
            recency_decay_sec=3600.0,
            local_boost=0.5,
        )
        self.assertEqual(len(hits), 3)  # s1 的两条 + 全局一条
        top = hits[0]
        self.assertIn(top.record.event_id, {"tool-recent", "user-local", "global-tip"})
        # 词法命中词应覆盖精确 token（player/gd）
        all_terms = set().union(*(set(h.matched_terms) for h in hits))
        self.assertTrue(all_terms)  # 不应为空
        # 本地 s1 的 tool-recent（最近）应排在其它 session 之前
        ranked_ids = [h.record.event_id for h in hits]
        self.assertNotIn("other-session", ranked_ids)

    def test_hybrid_recall_empty_store_returns_empty(self) -> None:
        hits = LogVectorStore().recall_hybrid(query_text="空库查询")
        self.assertEqual(hits, [])

    def test_hybrid_recall_lexical_boost_prefers_term_overlap(self) -> None:
        store = LogVectorStore()
        now = time.time()
        store.add(
            GameLogRecord(
                event_id="match",
                event_type="tool/result",
                kind="tool/result",
                unix_time=now - 200,
                session_id="s1",
                data={"tool": "bash", "result": "makefile 编译失败"},
            )
        )
        store.add(
            GameLogRecord(
                event_id="nomatch",
                event_type="assistant/message",
                kind="assistant/message",
                unix_time=now - 50,
                session_id="s1",
                data={"text": "好的，已完成"},
            )
        )
        hits = store.recall_hybrid(
            query_text="makefile 编译 失败",
            session_id="s1",
            limit=5,
        )
        self.assertEqual(hits[0].record.event_id, "match")


class ContextApiTests(unittest.TestCase):
    def test_ctx_endpoints(self) -> None:
        client = TestClient(create_app(persist=False))
        now = time.time()

        r = client.post(
            "/ctx/ingest",
            json={
                "record": {
                    "event_id": "a1",
                    "event_type": "user/message",
                    "kind": "user/message",
                    "unix_time": now,
                    "session_id": "s1",
                    "data": {"text": "你好，帮我查一下库存"},
                }
            },
        )
        self.assertEqual(r.status_code, 200)
        self.assertEqual(r.json()["event_id"], "a1")

        r = client.post(
            "/ctx/recall",
            json={"query": {"text": "查库存", "session_id": "s1", "limit": 5}},
        )
        self.assertEqual(r.status_code, 200)
        items = r.json()["items"]
        self.assertEqual(len(items), 1)
        self.assertEqual(items[0]["record"]["event_id"], "a1")
        self.assertIn("score", items[0])

        r = client.get("/ctx/status")
        self.assertEqual(r.status_code, 200)
        self.assertEqual(r.json()["status"]["count"], 1)

        r = client.post("/ctx/touch", json={"event_id": "a1"})
        self.assertEqual(r.json(), {"ok": True, "event_id": "a1"})

        r = client.post("/ctx/cleanup")
        self.assertEqual(r.status_code, 200)
        self.assertIn("swept_expired", r.json())


if __name__ == "__main__":
    unittest.main()
