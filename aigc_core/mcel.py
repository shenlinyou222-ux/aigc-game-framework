"""mini-CEL：手写的、参考 Google CEL 语义的极简表达式求值器（纯 Python）。

用途：把 graphrunner 里 CEL 那类“schema 校验/关系判断”降级成本地小 DSL，
校验 = 检索(resolvable) + 关系判断(relation)，不接任何外部接口。

支持（精简 CEL 子集）：
- 字面量：数字 / 字符串 / true / false / null
- 标识符与成员访问：player.location、data.items
- 列表/字典索引：items[0]、m["k"]
- 一元：! 、-
- 二元：算术 + - * / %；比较 == != < <= > >=；逻辑 && ||；in
- 函数调用（内置）：size / startsWith / endsWith / contains / matches / has
- 短路：&& / ||
用法：evaluate("size(items) > 0 && items[0].kind == 'ore'", {"items":[...]})
"""

from __future__ import annotations

import re
from typing import Any, Dict, List


# ---------------------------------------------------------------------------
# Tokenizer
# ---------------------------------------------------------------------------

_TOKEN_RE = re.compile(
    r"""
    (?P<WS>\s+)
  | (?P<NUM>\d+(?:\.\d+)?)
  | (?P<STR>'(?:[^'\\]|\\.)*'|"(?:[^"\\]|\\.)*")
  | (?P<OP>==|!=|<=|>=|&&|\|\||[+\-*/%<>!()\[\].,:])
  | (?P<ID>[A-Za-z_][A-Za-z0-9_]*)
    """,
    re.VERBOSE,
)

_KEYWORDS = {"true", "false", "null", "in"}


class _Token:
    __slots__ = ("kind", "value")

    def __init__(self, kind: str, value: Any) -> None:
        self.kind = kind
        self.value = value

    def __repr__(self) -> str:  # pragma: no cover - debug helper
        return f"{self.kind}:{self.value!r}"


def _tokenize(text: str) -> List[_Token]:
    tokens: List[_Token] = []
    pos = 0
    while pos < len(text):
        m = _TOKEN_RE.match(text, pos)
        if not m:
            raise ValueError(f"mcel: unexpected char at {pos}: {text[pos]!r}")
        pos = m.end()
        kind = m.lastgroup
        if kind == "WS":
            continue
        val = m.group()
        if kind == "NUM":
            tokens.append(_Token("NUM", float(val) if "." in val else int(val)))
        elif kind == "STR":
            tokens.append(_Token("STR", _unescape(val[1:-1])))
        elif kind == "ID":
            low = val
            if low in _KEYWORDS:
                tokens.append(_Token(low, low))
            else:
                tokens.append(_Token("ID", val))
        else:  # OP
            tokens.append(_Token("OP", val))
    tokens.append(_Token("EOF", ""))
    return tokens


def _unescape(s: str) -> str:
    return (
        s.replace("\\'", "'")
        .replace('\\"', '"')
        .replace("\\\\", "\\")
        .replace("\\n", "\n")
        .replace("\\t", "\t")
    )


# ---------------------------------------------------------------------------
# Parser -> AST
# ---------------------------------------------------------------------------


class _Parser:
    def __init__(self, tokens: List[_Token]) -> None:
        self.toks = tokens
        self.i = 0

    def peek(self) -> _Token:
        return self.toks[self.i]

    def next(self) -> _Token:
        t = self.toks[self.i]
        self.i += 1
        return t

    def expect_op(self, op: str) -> None:
        t = self.next()
        if t.kind != "OP" or t.value != op:
            raise ValueError(f"mcel: expected {op!r}, got {t.value!r}")

    def parse(self) -> Any:
        node = self._or()
        if self.peek().kind != "EOF":
            raise ValueError(f"mcel: trailing tokens near {self.peek().value!r}")
        return node

    def _or(self) -> Any:
        left = self._and()
        while self.peek().kind == "OP" and self.peek().value == "||":
            self.next()
            right = self._and()
            left = ("or", left, right)
        return left

    def _and(self) -> Any:
        left = self._cmp()
        while self.peek().kind == "OP" and self.peek().value == "&&":
            self.next()
            right = self._cmp()
            left = ("and", left, right)
        return left

    def _cmp(self) -> Any:
        left = self._add()
        t = self.peek()
        if t.kind == "OP" and t.value in ("==", "!=", "<", "<=", ">", ">="):
            self.next()
            right = self._add()
            return (t.value, left, right)
        if t.kind == "in":
            self.next()
            right = self._add()
            return ("in", left, right)
        return left

    def _add(self) -> Any:
        left = self._mul()
        while self.peek().kind == "OP" and self.peek().value in ("+", "-"):
            op = self.next().value
            right = self._mul()
            left = (op, left, right)
        return left

    def _mul(self) -> Any:
        left = self._unary()
        while self.peek().kind == "OP" and self.peek().value in ("*", "/", "%"):
            op = self.next().value
            right = self._unary()
            left = (op, left, right)
        return left

    def _unary(self) -> Any:
        t = self.peek()
        if t.kind == "OP" and t.value == "!":
            self.next()
            return ("not", self._unary())
        if t.kind == "OP" and t.value == "-":
            self.next()
            return ("neg", self._unary())
        return self._postfix()

    def _postfix(self) -> Any:
        node = self._primary()
        while True:
            t = self.peek()
            if t.kind == "OP" and t.value == ".":
                self.next()
                name = self.next()
                if name.kind != "ID":
                    raise ValueError(f"mcel: expected field after '.', got {name.value!r}")
                node = ("field", node, name.value)
            elif t.kind == "OP" and t.value == "[":
                self.next()
                idx = self._or()
                self.expect_op("]")
                node = ("index", node, idx)
            elif t.kind == "OP" and t.value == "(":
                node = self._call(node)
            else:
                break
        return node

    def _call(self, fn: Any) -> Any:
        self.expect_op("(")
        args: List[Any] = []
        if not (self.peek().kind == "OP" and self.peek().value == ")"):
            args.append(self._or())
            while self.peek().kind == "OP" and self.peek().value == ",":
                self.next()
                args.append(self._or())
        self.expect_op(")")
        return ("call", fn, args)

    def _primary(self) -> Any:
        t = self.next()
        if t.kind == "NUM":
            return ("lit", t.value)
        if t.kind == "STR":
            return ("lit", t.value)
        if t.kind == "true":
            return ("lit", True)
        if t.kind == "false":
            return ("lit", False)
        if t.kind == "null":
            return ("lit", None)
        if t.kind == "ID":
            return ("var", t.value)
        if t.kind == "OP" and t.value == "[":
            items: List[Any] = []
            if not (self.peek().kind == "OP" and self.peek().value == "]"):
                items.append(self._or())
                while self.peek().kind == "OP" and self.peek().value == ",":
                    self.next()
                    items.append(self._or())
            self.expect_op("]")
            return ("list", items)
        if t.kind == "OP" and t.value == "(":
            inner = self._or()
            self.expect_op(")")
            return inner
        raise ValueError(f"mcel: unexpected token {t.value!r}")


def parse(text: str) -> Any:
    return _Parser(_tokenize(text)).parse()


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------


class McelError(ValueError):
    pass


def _resolve_field(obj: Any, field: str) -> Any:
    if isinstance(obj, dict):
        if field not in obj:
            raise McelError(f"mcel: no field {field!r} in map")
        return obj[field]
    if hasattr(obj, field):
        return getattr(obj, field)
    raise McelError(f"mcel: cannot read field {field!r} from {type(obj).__name__}")


def _resolve_index(obj: Any, key: Any) -> Any:
    if isinstance(obj, list):
        if not isinstance(key, int):
            raise McelError("mcel: list index must be int")
        try:
            return obj[key]
        except IndexError:
            raise McelError(f"mcel: index {key} out of range") from None
    if isinstance(obj, dict):
        try:
            return obj[key]
        except KeyError:
            raise McelError(f"mcel: no key {key!r} in map") from None
    raise McelError(f"mcel: cannot index {type(obj).__name__}")


def _bin_math(op: str, a: Any, b: Any) -> Any:
    try:
        if op == "+":
            return a + b
        if op == "-":
            return a - b
        if op == "*":
            return a * b
        if op == "/":
            return a / b
        if op == "%":
            return a % b
    except Exception as exc:  # TypeError/ZeroDivision
        raise McelError(f"mcel: {op} failed on {a!r} and {b!r}: {exc}") from exc
    raise McelError(f"mcel: unknown op {op}")


def _contains(a: Any, b: Any) -> bool:
    if isinstance(a, (list, dict, str)):
        return b in a
    raise McelError(f"mcel: contains() not supported on {type(a).__name__}")


def _call_func(name: str, args: List[Any]) -> Any:
    if name == "size":
        if len(args) != 1:
            raise McelError("mcel: size() expects 1 arg")
        return len(args[0])
    if name == "startsWith":
        if len(args) != 2:
            raise McelError("mcel: startsWith() expects 2 args")
        return args[0].startswith(args[1])
    if name == "endsWith":
        if len(args) != 2:
            raise McelError("mcel: endsWith() expects 2 args")
        return args[0].endswith(args[1])
    if name == "contains":
        if len(args) != 2:
            raise McelError("mcel: contains() expects 2 args")
        return _contains(args[0], args[1])
    if name == "matches":
        if len(args) != 2:
            raise McelError("mcel: matches() expects 2 args")
        return re.search(args[1], args[0]) is not None
    if name == "has":
        if len(args) != 2:
            raise McelError("mcel: has() expects 2 args (map,key)")
        obj, key = args
        return isinstance(obj, dict) and key in obj
    raise McelError(f"mcel: unknown function {name}()")


class Evaluator:
    def __init__(self, context: Dict[str, Any]) -> None:
        self.ctx = context

    def eval(self, node: Any) -> Any:
        kind = node[0] if isinstance(node, tuple) else None
        if node[0] == "lit":
            return node[1]
        if node[0] == "var":
            name = node[1]
            if name not in self.ctx:
                raise McelError(f"mcel: undefined variable {name!r}")
            return self.ctx[name]
        if node[0] == "field":
            base = self.eval(node[1])
            return _resolve_field(base, node[2])
        if node[0] == "index":
            base = self.eval(node[1])
            key = self.eval(node[2])
            return _resolve_index(base, key)
        if node[0] == "list":
            return [self.eval(item) for item in node[1]]
        if node[0] == "not":
            return not self._bool(self.eval(node[1]))
        if node[0] == "neg":
            return -self.eval(node[1])
        if node[0] == "or":
            left = self._bool(self.eval(node[1]))
            return left or self._bool(self.eval(node[2]))
        if node[0] == "and":
            left = self._bool(self.eval(node[1]))
            return left and self._bool(self.eval(node[2]))
        if node[0] in ("==", "!=", "<", "<=", ">", ">="):
            a, b = self.eval(node[1]), self.eval(node[2])
            return _compare(node[0], a, b)
        if node[0] == "in":
            a, b = self.eval(node[1]), self.eval(node[2])
            return _contains(b, a)
        if node[0] in ("+", "-", "*", "/", "%"):
            return _bin_math(node[0], self.eval(node[1]), self.eval(node[2]))
        if node[0] == "call":
            fn_node = node[1]
            if fn_node[0] == "var":
                name = fn_node[1]
            else:
                raise McelError("mcel: only direct function names supported")
            args = [self.eval(a) for a in node[2]]
            return _call_func(name, args)
        raise McelError(f"mcel: unknown ast node {node!r}")

    @staticmethod
    def _bool(v: Any) -> bool:
        if not isinstance(v, bool):
            raise McelError(f"mcel: expected bool, got {type(v).__name__}: {v!r}")
        return v


def _compare(op: str, a: Any, b: Any) -> bool:
    if op == "==":
        return _eq(a, b)
    if op == "!=":
        return not _eq(a, b)
    # CEL 不允许数字/字符串混比，这里做温和归一
    try:
        if op == "<":
            return a < b
        if op == "<=":
            return a <= b
        if op == ">":
            return a > b
        if op == ">=":
            return a >= b
    except Exception as exc:
        raise McelError(f"mcel: compare {op} failed: {exc}") from exc
    raise McelError(f"mcel: unknown compare {op}")


def _eq(a: Any, b: Any) -> bool:
    if isinstance(a, bool) != isinstance(b, bool):
        return False
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        return a == b
    return type(a) is type(b) and a == b


def evaluate(expr: str, context: Dict[str, Any]) -> Any:
    """解析并求值一条 mini-CEL 表达式。"""
    return Evaluator(context).eval(parse(expr))
