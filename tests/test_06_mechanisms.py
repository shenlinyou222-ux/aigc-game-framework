"""06 机制单测：query_normalize / fact 层 / tool 层 / WorkingSet / planner / /ctx/* API。

覆盖设计文档 06 的核心承诺：
- LLM=意图+取舍；查询轻量清洗后原样进库（不重写语义）
- 事实/事件/工具同构，可一并查出
- WorkingSet manifest：只记 id/kind/chars，load 先过预算再展开
- 程序负责组装：messages 里“整段历史重放段”替换成 WorkingSet 展开 + 工具 spec + 本轮请求
- 留 search/load/drop 口子（InternalFunction 注册）
"""

from __future__ import annotations

import unittest

from fastapi.testclient import TestClient

from aigc_core import planner
from aigc_core.api import create_app
from aigc_core.functions import FunctionRegistry
from aigc_core.ctx_tools import register_context_tools
from aigc_core.query_normalize import QueryNormalizer, query_normalize
from aigc_core.working_set import BudgetError, WorkingSetManifest
from aigc_core.interface_store import InterfaceSpec


def _client() -> TestClient:
    return TestClient(create_app(persist=False))


class QueryNormalizeTests(unittest.TestCase):
    def test_cleans_fillers_but_keeps_raw_semantics(self) -> None:
        out = query_normalize("麻烦帮我查一下 player.gd 的报错，顺便看看库存")
        # 去停用词但保留原始关键词/子句（不改写语义）
        self.assertTrue(out)
        joined = " ".join(out).lower()
        self.assertIn("player.gd", joined)
        self.assertIn("报错", joined)
        self.assertIn("库存", joined)

    def test_splits_into_independent_raw_clauses(self) -> None:
        n = QueryNormalizer(max_clauses=3)
        out = n.normalize("帮我查库存；然后找一下背包里的铜矿")
        self.assertLessEqual(len(out), 3)
        self.assertGreaterEqual(len(out), 2)

    def test_empty_query_returns_empty(self) -> None:
        self.assertEqual(query_normalize(""), [])


class WorkingSetTests(unittest.TestCase):
    def test_budget_count_and_revision(self) -> None:
        m = WorkingSetManifest(session="s1", max_count=1, max_tokens=100)
        m.check_add("a", 10)
        m.add_item("a", kind="event", chars=10)
        self.assertEqual(m.count(), 1)
        with self.assertRaises(BudgetError):
            m.check_add("b", 10)  # 超过 maxCount
        self.assertEqual(m.revision, 1)


class PlannerTests(unittest.TestCase):
    def test_parse_plan_accepts_structured_json(self) -> None:
        plan = planner.parse_plan(
            '{"intent":"recall","scope":["fact","event","tool"],'
            '"queries":["查库存"],"load":["fact/a1"],"drop":[]}'
        )
        self.assertEqual(plan.intent, "recall")
        self.assertEqual(plan.load, ["fact/a1"])

    def test_parse_plan_rejects_extra_keys(self) -> None:
        with self.assertRaises(ValueError):
            planner.parse_plan('{"intent":"recall","queries":[],"essay":"自由长文本"}')

    def test_assemble_messages_replaces_history_replay(self) -> None:
        msgs = planner.assemble_messages(
            [{"id": "fact/a1", "kind": "fact", "text": "玩家喜欢种萝卜"}],
            user_request="帮我种萝卜",
            tool_specs=[{"name": "give_item"}],
        )
        types = [m["type"] for m in msgs if m.get("type")]
        self.assertIn("working_set", types)
        self.assertIn("tool_spec", types)
        self.assertEqual(msgs[-1], {"role": "user", "content": "帮我种萝卜"})


class ControllerMechanismTests(unittest.TestCase):
    def test_fact_tool_event_unified_recall_and_assemble(self) -> None:
        client = _client()
        # 事实层：写压缩 + 搜索
        r1 = client.post("/ctx/fact/ingest", json={"text": "玩家最喜欢种萝卜并收藏铜矿"})
        self.assertEqual(r1.status_code, 200)
        f1 = r1.json()["record"]
        # 同文本再入库应去重（写压缩）
        r2 = client.post("/ctx/fact/ingest", json={"text": "玩家最喜欢种萝卜并收藏铜矿"})
        self.assertEqual(r2.json()["record"]["event_id"], f1["event_id"])

        # 事件层
        client.post(
            "/ctx/ingest",
            json={
                "record": {
                    "event_id": "ev-萝卜",
                    "event_type": "user/message",
                    "kind": "user/message",
                    "session_id": "s1",
                    "data": {"text": "这轮把萝卜种到田里"},
                }
            },
        )
        # 工具层（复用 default interface specs）
        r = client.post("/ctx/tool/search", json={"text": "创建一种新的物品种类", "top_k": 3})
        self.assertEqual(r.status_code, 200)
        tool_items = r.json()["items"]
        self.assertTrue(any(i["id"] == "tool/create_item" for i in tool_items))

        # fact search
        r = client.post("/ctx/fact/search", json={"text": "种萝卜 收藏 铜矿", "limit": 3})
        self.assertTrue(r.json()["items"])

        # working set load/expand + assemble
        r = client.post("/ctx/working-set", json={"session": "s1"})
        self.assertEqual(r.status_code, 200)
        snap = r.json()["manifest"]
        self.assertEqual(snap["count"], 0)

        r = client.post(
            "/ctx/working-set/load",
            json={"session": "s1", "ids": [{"id": f1["event_id"], "kind": "fact"}]},
        )
        self.assertEqual(r.status_code, 200)
        self.assertEqual(len(r.json()["loaded"]), 1)

        r = client.post("/ctx/working-set/expand", json={"session": "s1"})
        self.assertEqual(len(r.json()["items"]), 1)
        self.assertEqual(r.json()["items"][0]["kind"], "fact")

        # assemble：返回 messages，历史重放段已被 WorkingSet 展开替换
        r = client.post(
            "/ctx/assemble",
            json={"session": "s1", "user_text": "帮我种萝卜，顺便看看要什么工具", "auto_load": True},
        )
        self.assertEqual(r.status_code, 200)
        body = r.json()
        self.assertTrue(body["ok"])
        self.assertIn("clauses", body)
        self.assertTrue(body["working_set"]["count"] >= 1)
        self.assertTrue(body["messages"])  # working_set 展开 + 本轮请求

    def test_ctx_tools_registered_as_internal_functions(self) -> None:
        from aigc_core import MainController

        ctrl = MainController()
        reg = FunctionRegistry()
        register_context_tools(reg, ctrl, session="s1")
        names = reg.names()
        self.assertIn("search_context", names)
        self.assertIn("load_context", names)
        self.assertIn("drop_context", names)


if __name__ == "__main__":
    unittest.main()
