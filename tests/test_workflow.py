"""工作流 JSON 解释器测试：串行 call / map 循环 / if 条件 / emit 收集。"""

from __future__ import annotations

import unittest

from aigc_core.workflow import WorkflowError, run_workflow


def _handlers():
    return {
        "query_items": lambda _in: {"items": [
            {"id": "a", "kind": "copper"},
            {"id": "b", "kind": "iron"},
            {"id": "c", "kind": "gold"},
        ]},
        "make_item": lambda in_: {"item_id": in_["item"]["id"], "kind": in_["item"]["kind"]},
        "play_music": lambda in_: {"track": in_["track"]},
    }


class WorkflowTests(unittest.TestCase):
    def test_serial_call_and_map_with_if(self) -> None:
        spec = {
            "input": {"token": "T1"},
            "steps": [
                {"id": "s1", "op": "call", "interface": "query_items", "in": {}, "into": "items_res"},
                {
                    "id": "s2", "op": "map", "over": "$items_res.items", "as": "it", "into": "made_list",
                    "do": [
                        {
                            "id": "s2a", "op": "if",
                            "when": "it.kind in ['copper', 'iron']",
                            "then": [
                                {"id": "s2a1", "op": "call", "interface": "make_item",
                                 "in": {"item": "$it", "token": "$token"}, "into": "one"},
                                {"id": "s2a2", "op": "emit", "value": "$one"},
                            ],
                            "else": [],
                        }
                    ],
                },
                {
                    "id": "s3", "op": "if", "when": "size(made_list) > 0",
                    "then": [
                        {"id": "s3a", "op": "call", "interface": "play_music",
                         "in": {"track": "rain"}, "into": "music"}
                    ],
                },
            ],
            "output": ["made_list", "music"],
        }
        env = run_workflow(spec, _handlers())
        # 只有 copper/iron 通过 if，gold 被过滤
        self.assertEqual(
            [r["item_id"] for r in env["made_list"]],
            ["a", "b"],
        )
        self.assertEqual(env["music"]["track"], "rain")
        self.assertEqual(env["outputs"]["made_list"], env["made_list"])

    def test_empty_list_map_yields_empty_and_skips(self) -> None:
        spec = {
            "steps": [
                {"id": "s1", "op": "call", "interface": "query_items", "in": {}, "into": "items_res"},
                {"id": "s2", "op": "map", "over": "$items_res.items", "as": "it", "into": "made_list",
                 "do": [{"id": "s2a", "op": "call", "interface": "make_item",
                         "in": {"item": "$it"}, "into": "one"},
                        {"id": "s2b", "op": "emit", "value": "$one"}]},
            ],
            "output": ["made_list"],
        }
        handlers = {"query_items": lambda _in: {"items": []}, "make_item": lambda _in: {}}
        env = run_workflow(spec, handlers)
        self.assertEqual(env["made_list"], [])
        self.assertEqual(env["outputs"]["made_list"], [])

    def test_unknown_op_raises(self) -> None:
        with self.assertRaises(WorkflowError):
            run_workflow({"steps": [{"op": "nope"}]}, {})

    def test_unknown_interface_raises(self) -> None:
        with self.assertRaises(WorkflowError):
            run_workflow(
                {"steps": [{"op": "call", "interface": "ghost", "in": {}, "into": "x"}]},
                {},
            )


if __name__ == "__main__":
    unittest.main()
