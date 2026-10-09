"""functions.py —— 内部函数地基。

编排最终 = LLM 用“内部函数”写主函数并执行；这里补的是内部函数本身的定义层：
1. 每个内部函数带 schema(input_fields/env_fields) + CEL 前置规则(rules)；
2. 调用时用 `env`（可每次不同）+ `params` 本地组装 context，不调其它服务端；
3. 命中规则 -> 抛结构化 `RuleError(func, rule, expr, ctx)`，可精确到“撞了哪条规则”；
4. 下游函数抛的 RuleError 会原样透传（不吞、不重包），顶层永远知道在哪层撞了哪条规则。
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

from . import mcel

# 内部函数实现：吃一个合并好的 context(=env+params) -> JSON 输出
Impl = Callable[[Dict[str, Any]], Dict[str, Any]]


@dataclass
class Rule:
    id: str                 # 规则名，例如 "actor_allowed"
    expr: str               # mini-CEL 表达式（变量 = context 的扁平键）
    message: str = ""


class RuleError(RuntimeError):
    """规则级错误：到底哪个函数、撞了哪条 CEL 规则、当时 context 是什么。"""

    def __init__(
        self,
        func: str,
        rule: str,
        expr: str,
        context: Dict[str, Any],
        cause: Optional[BaseException] = None,
        message: str = "",
    ) -> None:
        self.func = func
        self.rule = rule
        self.expr = expr
        self.context = dict(context)
        self.cause = cause
        detail = message or f"{func}.{rule} failed"
        super().__init__(f"[{detail}] expr={expr} ctx={json.dumps(self.context, ensure_ascii=False, default=str)[:400]}")


@dataclass
class InternalFunction:
    name: str
    description: str
    fn: Impl
    input_fields: Dict[str, str] = field(default_factory=dict)  # 入参 schema：字段名 -> 说明
    env_fields: List[str] = field(default_factory=list)          # 它用到的 env 字段名
    rules: List[Rule] = field(default_factory=list)              # CEL 前置规则
    output_schema: Dict[str, Any] = field(default_factory=dict)

    def call(self, params: Dict[str, Any], env: Dict[str, Any]) -> Dict[str, Any]:
        return call_internal(self, params, env)


def call_internal(fn: InternalFunction, params: Dict[str, Any], env: Dict[str, Any]) -> Dict[str, Any]:
    """统一调用入口：本地组装 context -> 逐个过 CEL 规则 -> 执行。

    - 上下文只来自 env + params（本地），不调其它服务端接口。
    - 命中规则 -> RuleError。
    - fn 内部调用下游函数若抛 RuleError -> 原样透传。
    """
    context: Dict[str, Any] = dict(env)
    context.update(params)
    for rule in fn.rules:
        ok = mcel.evaluate(rule.expr, context)
        if ok is not True:
            raise RuleError(fn.name, rule.id, rule.expr, context, message=rule.message)
    try:
        return fn.fn(context)
    except RuleError as exc:
        raise  # 下游 RuleError 透传：不吞、不重包
    except Exception as exc:
        raise RuleError(fn.name, "runtime", "fn(...)", context, cause=exc, message=exc.__class__.__name__)


class FunctionRegistry:
    """按名字注册/查找内部函数。"""

    def __init__(self) -> None:
        self._fns: Dict[str, InternalFunction] = {}

    def register(self, fn: InternalFunction) -> "FunctionRegistry":
        self._fns[fn.name] = fn
        return self

    def get(self, name: str) -> Optional[InternalFunction]:
        return self._fns.get(name)

    def names(self) -> List[str]:
        return sorted(self._fns.keys())

    def call(self, name: str, params: Dict[str, Any], env: Dict[str, Any]) -> Dict[str, Any]:
        fn = self.get(name)
        if fn is None:
            raise RuleError(name, "unknown", "", {"env": env, "params": params}, message="unknown function")
        return call_internal(fn, params, env)
