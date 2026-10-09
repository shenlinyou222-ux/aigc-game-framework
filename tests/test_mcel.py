"""mini-CEL 求值器测试。"""

from __future__ import annotations

import unittest

from aigc_core.mcel import McelError, evaluate


class MiniCelTests(unittest.TestCase):
    def test_literals_and_arithmetic(self) -> None:
        self.assertTrue(evaluate("1 + 2 * 3 == 7", {}))
        self.assertFalse(evaluate("(1 + 2) * 3 == 7", {}))

    def test_compare_and_logic_short_circuit(self) -> None:
        self.assertTrue(evaluate("2 < 3 && 3 <= 3 && 4 > 3 && 4 >= 4", {}))
        # 短路：true || 右边不 eval（右边若报错也不影响）
        self.assertTrue(evaluate("true || undefined_var == 1", {}))
        # false && 右边不 eval，整体为 false
        self.assertFalse(evaluate("false && undefined_var == 1", {}))

    def test_member_and_index(self) -> None:
        ctx = {"player": {"location": "farm", "items": [{"kind": "ore", "n": 2}]}}
        self.assertEqual(evaluate("player.location", ctx), "farm")
        self.assertEqual(evaluate("player.items[0].kind", ctx), "ore")
        self.assertEqual(evaluate("player.items[0].n + 1", ctx), 3)

    def test_list_literal(self) -> None:
        self.assertTrue(evaluate("'copper' in ['copper', 'iron']", {}))
        self.assertTrue(evaluate("kind in ['copper', 'iron']", {"kind": "copper"}))
        self.assertEqual(evaluate("[1, 2, 3][1]", {}), 2)

    def test_in_and_contains(self) -> None:
        ctx = {
            "loc": "farm",
            "allowed": ["farm", "village"],
            "text": "hello world",
            "m": {"a": 1},
        }
        self.assertTrue(evaluate("loc in allowed", ctx))
        self.assertTrue(evaluate("contains(text, 'world')", ctx))
        self.assertTrue(evaluate("has(m, 'a')", ctx))
        self.assertFalse(evaluate("has(m, 'b')", ctx))

    def test_string_funcs(self) -> None:
        self.assertTrue(evaluate("startsWith('apple.png', 'apple')", {}))
        self.assertTrue(evaluate("endsWith('apple.png', '.png')", {}))
        self.assertTrue(evaluate("size('abc') == 3", {}))
        self.assertTrue(evaluate("matches('abc123', '\\\\d+')", {}))

    def test_null_and_neg(self) -> None:
        self.assertTrue(evaluate("!(1 == 2)", {}))
        self.assertTrue(evaluate("-3 < 0", {}))
        self.assertTrue(evaluate("null == null", {}))

    def test_missing_field_raises(self) -> None:
        with self.assertRaises(McelError):
            evaluate("player.ghost", {"player": {}})

    def test_bad_type_raises(self) -> None:
        with self.assertRaises(McelError):
            evaluate("'a' && true", {})


if __name__ == "__main__":
    unittest.main()
