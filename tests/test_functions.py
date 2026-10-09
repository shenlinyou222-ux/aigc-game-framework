"""内部函数地基测试：env 切换 / CEL 规则级报错 / 下游 RuleError 透传 / 本地上下文组装。"""

from __future__ import annotations

import unittest

from aigc_core.functions import (
    FunctionRegistry,
    InternalFunction,
    Rule,
    RuleError,
)


class FunctionBaseTests(unittest.TestCase):
    def test_env_switching_hits_actor_rule(self) -> None:
        fn = InternalFunction(
            name="play_music",
            description="切歌",
            env_fields=["actor"],
            rules=[Rule(id="actor_allowed", expr="actor in ['system', 'assistant']")],
            fn=lambda ctx: {"track": ctx["track"]},
        )
        reg = FunctionRegistry().register(fn)

        # assistant 允许
        self.assertEqual(
            reg.call("play_music", {"track": "rain"}, {"actor": "assistant"})["track"],
            "rain",
        )
        # 换 env：robot 不被允许 -> 精确命中 actor_allowed
        with self.assertRaises(RuleError) as cm:
            reg.call("play_music", {"track": "rain"}, {"actor": "robot"})
        self.assertEqual(cm.exception.func, "play_music")
        self.assertEqual(cm.exception.rule, "actor_allowed")
        self.assertIn("actor_allowed", str(cm.exception))

    def test_local_context_assembly_catalog(self) -> None:
        # 校验=检索+关系判断：catalog 由调用方本地组装进 env，不调其它服务端
        fn = InternalFunction(
            name="give_item",
            description="发物品",
            env_fields=["actor", "catalog"],
            input_fields={"item_id": "物品id"},
            rules=[Rule(id="item_exists", expr="item_id in catalog")],
            fn=lambda ctx: {"ok": True, "item_id": ctx["item_id"]},
        )
        reg = FunctionRegistry().register(fn)
        env = {"actor": "system", "catalog": ["test_item", "wood"]}

        self.assertTrue(reg.call("give_item", {"item_id": "test_item"}, env)["ok"])
        with self.assertRaises(RuleError) as cm:
            reg.call("give_item", {"item_id": "ghost"}, env)
        self.assertEqual(cm.exception.rule, "item_exists")

    def test_downstream_rule_error_passthrough(self) -> None:
        inner = InternalFunction(
            name="inner",
            description="内部下游",
            rules=[Rule(id="always_false", expr="x in ['never']")],
            fn=lambda ctx: {"ok": True},
        )
        outer_reg = FunctionRegistry().register(inner)

        def outer_impl(ctx):
            # 外层调用下游 inner -> 抛 RuleError，必须原样透传
            return outer_reg.call("inner", {"x": 1}, {"actor": "system"})

        outer = InternalFunction(
            name="outer",
            description="外层",
            rules=[Rule(id="outer_pre", expr="true")],
            fn=outer_impl,
        )
        reg = FunctionRegistry().register(outer).register(inner)
        with self.assertRaises(RuleError) as cm:
            reg.call("outer", {}, {"actor": "system"})
        # 不是被包成 runtime，规则信息保留的是 inner 的
        self.assertEqual(cm.exception.func, "inner")
        self.assertEqual(cm.exception.rule, "always_false")
        self.assertNotEqual(cm.exception.rule, "runtime")

    def test_runtime_error_wrapped_as_rule_error(self) -> None:
        fn = InternalFunction(
            name="boom",
            description="会崩",
            fn=lambda ctx: (_ for _ in ()).throw(ValueError("boom")),
        )
        reg = FunctionRegistry().register(fn)
        with self.assertRaises(RuleError) as cm:
            reg.call("boom", {}, {})
        self.assertEqual(cm.exception.rule, "runtime")
        self.assertIsInstance(cm.exception.cause, ValueError)

    def test_unknown_function(self) -> None:
        with self.assertRaises(RuleError) as cm:
            FunctionRegistry().call("ghost", {}, {})
        self.assertEqual(cm.exception.rule, "unknown")


if __name__ == "__main__":
    unittest.main()
