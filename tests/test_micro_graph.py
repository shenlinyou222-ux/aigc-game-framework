"""micro-graph 编排内核测试：绑定 / 执行 / 校验 / 预检 / 错误上下文。"""

from __future__ import annotations

import asyncio
import unittest

from aigc_core.micro_graph import (
    Binding,
    GraphSpec,
    NodeError,
    NodeSpec,
    arun,
    expand_graph,
    graph_from_dict,
    graph_to_dict,
    navigate,
    preflight,
    run,
)


def _graph() -> GraphSpec:
    return GraphSpec(
        nodes=[
            NodeSpec(key="pick", name="pick_ore"),
            NodeSpec(key="make", name="make_item"),
        ],
        bindings=[
            Binding(from_node="START", from_field="player.location", to_node="pick", to_field="loc"),
            Binding(from_node="pick", from_field="Data.items[0].kind", to_node="make", to_field="item_kind"),
            Binding(from_node="START", from_field="name", to_node="make", to_field="name"),
        ],
        input={
            "player": {"location": "mine"},
            "name": "test",
        },
    )


class NavigateTests(unittest.TestCase):
    def test_nested_and_index(self) -> None:
        data = {"Data": {"ClusterSet": [{"ClusterId": "c1"}]}}
        self.assertEqual(navigate(data, "Data.ClusterSet[0].ClusterId"), "c1")

    def test_missing_raises(self) -> None:
        with self.assertRaises(KeyError):
            navigate({"a": 1}, "a.b")


class RunnerTests(unittest.TestCase):
    def test_run_binds_upstream_output(self) -> None:
        def pick_ore(inp):
            self.assertEqual(inp["loc"], "mine")
            return {"Data": {"items": [{"kind": "copper"}]}}

        def make_item(inp):
            self.assertEqual(inp["item_kind"], "copper")
            self.assertEqual(inp["name"], "test")
            return {"item_id": "test_item"}

        res = run(
            _graph(),
            {"pick_ore": pick_ore, "make_item": make_item},
            rules={"make": "item_kind in ['copper', 'iron'] && name == 'test'"},
        )
        self.assertTrue(res.ok)
        self.assertEqual(res.outputs["make"]["item_id"], "test_item")
        self.assertEqual([s.status for s in res.steps], ["success", "success"])

    def test_rule_violation_fails_at_that_node_with_log(self) -> None:
        def pick_ore(inp):
            return {"Data": {"items": [{"kind": "gold"}]}}

        def make_item(inp):
            return {"ok": True}

        rules = {"make": "item_kind in ['copper', 'iron'] && name == 'test'"}
        with self.assertRaises(NodeError) as cm:
            run(_graph(), {"pick_ore": pick_ore, "make_item": make_item}, rules=rules)
        self.assertEqual(cm.exception.node_key, "make")
        self.assertEqual(cm.exception.log[-1].status, "failed")
        # 前面成功的 step 仍在日志里（上下文不丢）
        self.assertEqual([s.key for s in cm.exception.log], ["pick", "make"])

    def test_async_run(self) -> None:
        async def pick_ore(inp):
            await asyncio.sleep(0)
            return {"Data": {"items": [{"kind": "copper"}]}}

        def make_item(inp):
            return {"item_id": "async_test"}

        async def main():
            return await arun(_graph(), {"pick_ore": pick_ore, "make_item": make_item})

        res = asyncio.run(main())
        self.assertTrue(res.ok)
        self.assertEqual(res.outputs["make"]["item_id"], "async_test")


class PreflightTests(unittest.TestCase):
    def test_preflight_binds_and_validates_without_running(self) -> None:
        templates = {"pick": {"Data": {"items": [{"kind": "copper"}]}}}
        rules = {"make": "item_kind == 'copper' && name == 'test'"}
        bound = preflight(_graph(), rules=rules, output_templates=templates)
        self.assertEqual(bound["pick"]["loc"], "mine")
        self.assertEqual(bound["make"]["item_kind"], "copper")
        self.assertEqual(bound["make"]["name"], "test")

    def test_preflight_rule_failure_raises(self) -> None:
        templates = {"pick": {"Data": {"items": [{"kind": "gold"}]}}}
        rules = {"make": "item_kind in ['copper','iron']"}
        with self.assertRaises(Exception):
            preflight(_graph(), rules=rules, output_templates=templates)


class GraphExpansionTests(unittest.TestCase):
    """图展开：上游返回列表 -> 模板节点 unroll 成 N 个具体节点(以图 JSON 表达)。"""

    def _template_graph(self) -> GraphSpec:
        # 声明式：call 是“展开模板”，over=src.items 里的每个元素调一次 make_item
        return GraphSpec(
            nodes=[
                NodeSpec(key="src", name="list_items"),
                NodeSpec(
                    key="call",
                    name="make_item",
                    expand={"over": "src", "field": "Data.items", "as": "item", "key_field": "id"},
                ),
            ],
            bindings=[
                Binding(from_node="START", from_field="token", to_node="call", to_field="token"),
            ],
            input={"token": "T-1"},
        )

    def test_expand_unrolls_three_elements_into_nodes(self) -> None:
        spec = self._template_graph()
        out = {"src": {"Data": {"items": [
            {"id": "a", "kind": "copper"},
            {"id": "b", "kind": "iron"},
            {"id": "c", "kind": "gold"},
        ]}}}
        expanded = expand_graph(spec, out)
        keys = [n.key for n in expanded.nodes]
        self.assertIn("src", keys)
        self.assertIn("call_a", keys)
        self.assertIn("call_b", keys)
        self.assertIn("call_c", keys)
        self.assertNotIn("call", keys)  # 模板被移除

        # 每个展开节点都有“从 src.Data.items[N] 取元素”的索引边 + 常量 token 边
        for suffix, idx in (("a", 0), ("b", 1), ("c", 2)):
            binds = [b for b in expanded.bindings if b.to_node == f"call_{suffix}"]
            self.assertEqual(len(binds), 2)
            self.assertTrue(any(b.from_node == "src" and b.from_field == f"Data.items[{idx}]" for b in binds))
            self.assertTrue(any(b.from_node == "START" and b.to_field == "token" for b in binds))

    def test_expanded_graph_runs_each_element(self) -> None:
        spec = self._template_graph()
        out = {"src": {"Data": {"items": [
            {"id": "a", "kind": "copper"},
            {"id": "b", "kind": "iron"},
        ]}}}
        expanded = expand_graph(spec, out)

        def list_items(_in):
            return {"Data": {"items": [{"id": "a", "kind": "copper"}, {"id": "b", "kind": "iron"}]}}

        seen = {}

        def make_item(in_):
            item = in_["item"]
            seen[item["kind"]] = in_["token"]
            return {"ok": True, "kind": item["kind"]}

        res = run(expanded, {"list_items": list_items, "make_item": make_item})
        self.assertTrue(res.ok)
        self.assertEqual(set(seen), {"copper", "iron"})
        self.assertEqual(seen["copper"], "T-1")  # 常量 token 复制到每个展开节点

    def test_empty_list_expands_to_zero_nodes(self) -> None:
        spec = self._template_graph()
        expanded = expand_graph(spec, {"src": {"Data": {"items": []}}})
        keys = [n.key for n in expanded.nodes]
        self.assertEqual(keys, ["src"])  # 模板移除，无 call_*
        res = run(expanded, {"list_items": lambda _i: {"Data": {"items": []}}, "make_item": lambda _i: {}})
        self.assertTrue(res.ok)
        self.assertEqual(len(res.steps), 1)  # 只有 src，没调接口

    def test_graph_json_roundtrip_with_expand(self) -> None:
        spec = self._template_graph()
        d = graph_to_dict(spec)
        back = graph_from_dict(d)
        self.assertEqual(graph_to_dict(back), d)
        self.assertIsNotNone(back.nodes[1].expand)
        out = {"src": {"Data": {"items": [{"id": "a", "kind": "copper"}]}}}
        expanded_dict = graph_to_dict(expand_graph(back, out))
        self.assertEqual(expanded_dict["nodes"][1]["key"], "call_a")


if __name__ == "__main__":
    unittest.main()
