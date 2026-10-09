"""受控执行器测试：语法校验 / 白名单 / fallback 逻辑。"""

from __future__ import annotations

import unittest

from aigc_core.functions import FunctionRegistry, InternalFunction, Rule, RuleError
from aigc_core.python_runner import PythonRunnerError, run_python


def _registry() -> FunctionRegistry:
    reg = FunctionRegistry()

    reg.register(InternalFunction(
        name="query_items",
        description="查询",
        rules=[Rule(id="actor_ok", expr="actor in ['system','assistant']")],
        fn=lambda ctx: {"items": [{"kind": "copper"}, {"kind": "iron"}, {"kind": "gold"}]},
    ))
    reg.register(InternalFunction(
        name="make_item",
        description="造物",
        rules=[Rule(id="kind_ok", expr="kind in ['copper','iron']")],
        fn=lambda ctx: {"made": ctx["kind"]},
    ))
    return reg


class PythonRunnerTests(unittest.TestCase):
    def test_valid_main_runs(self) -> None:
        body = (
            "def main(ctx):\n"
            "    items = run('query_items', {})['items']\n"
            "    return [run('make_item', {'kind': it['kind']}) for it in items\n"
            "            if it['kind'] in ('copper', 'iron')]\n"
        )
        res = run_python(body, _registry(), env={"actor": "assistant"})
        self.assertEqual([r["made"] for r in res], ["copper", "iron"])

    def test_import_denied(self) -> None:
        with self.assertRaises(PythonRunnerError):
            run_python("import os\ndef main(ctx): return {}\n", _registry())

    def test_attribute_access_denied(self) -> None:
        with self.assertRaises(PythonRunnerError):
            run_python("def main(ctx): return os.system\n", _registry())

    def test_unknown_call_denied(self) -> None:
        with self.assertRaises(PythonRunnerError):
            run_python("def main(ctx): return eval('1+1')\n", _registry())

    def test_syntax_error_fallback(self) -> None:
        body = "def main(ctx):\n  return \n"  # 语法其实合法；造一个真语法错
        body = "def main(ctx):\n  return (1\n"  # 缺右括号
        res = run_python(body, _registry(), env={}, on_error=lambda e: {"fallback": str(e)})
        self.assertIn("fallback", res)

    def test_rule_error_fallback(self) -> None:
        body = (
            "def main(ctx):\n"
            "    return run('make_item', {'kind': 'gold'})\n"  # 命中 kind_ok 规则
        )
        captured = {}

        def on_error(e):
            captured["err"] = e
            return {"fallback": True}

        res = run_python(body, _registry(), env={"actor": "assistant"}, on_error=on_error)
        self.assertTrue(res["fallback"])
        self.assertIsInstance(captured["err"], RuleError)
        self.assertEqual(captured["err"].rule, "kind_ok")

    def test_no_fallback_raises(self) -> None:
        body = "def main(ctx):\n    return run('make_item', {'kind':'gold'})\n"
        with self.assertRaises(RuleError) as cm:
            run_python(body, _registry(), env={"actor": "assistant"})
        self.assertEqual(cm.exception.rule, "kind_ok")

    def test_missing_main_raises(self) -> None:
        with self.assertRaises(PythonRunnerError):
            run_python("x = 1\n", _registry())


if __name__ == "__main__":
    unittest.main()
