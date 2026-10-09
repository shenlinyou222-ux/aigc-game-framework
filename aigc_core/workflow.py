"""workflow.py —— “主函数”式 JSON 工作流解释器。

思路：把编排退化成“主函数”。LLM/上层输出一份层次清楚的 JSON：
- 顶层永远是一个扁平的 `steps` 列表（串行，先到先执行）
- `call`   = 调一个内部函数（纯函数）
- `map`    = 循环：over 一个列表，as 元素变量，do 一段 `steps`，emit 收集结果
- `if`     = 分支：when 用 mini-CEL 判断，then/else 各是一段 `steps`
- `assign` = 存一个值/引用
控制层次只体现在 map/if 的 do/then/else 里再放一段“同样扁平的 steps”，
不搞疯狂嵌套。执行器把 JSON 直接翻成 Python 函数调用（无图、无展开）。
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional

from . import mcel
from .micro_graph import navigate


class WorkflowError(RuntimeError):
    pass


# 内部函数：JSON in -> JSON out（纯函数；副作用只留到最外层执行器）
Handler = Callable[[Dict[str, Any]], Dict[str, Any]]


def _resolve(value: Any, env: Dict[str, Any]) -> Any:
    """$a.b.c 表示从环境里按路径取值；其余按字面量。"""
    if isinstance(value, str) and value.startswith("$"):
        return navigate(env, value[1:])
    if isinstance(value, dict):
        return {k: _resolve(v, env) for k, v in value.items()}
    if isinstance(value, list):
        return [_resolve(v, env) for v in value]
    return value


def run_workflow(
    spec: Dict[str, Any],
    handlers: Dict[str, Handler],
    input_data: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """执行一份工作流 JSON。返回：{...环境变量, "outputs": {output名: 值}}。"""
    top_input: Dict[str, Any] = dict(spec.get("input") or {})
    if input_data:
        top_input.update(input_data)
    env: Dict[str, Any] = {"input": top_input}
    env.update(top_input)  # 顶层输入也直接暴露成变量名，方便 $ref / mcel

    collected_here: List[Any] = []
    _run_steps(spec.get("steps", []), handlers, env, collected_here)
    env["outputs"] = {name: _resolve_ref(name, env) for name in (spec.get("output") or [])}
    return env


def _resolve_ref(name: str, env: Dict[str, Any]) -> Any:
    try:
        return navigate(env, name)
    except Exception:
        return None


def _run_steps(
    steps: List[Dict[str, Any]],
    handlers: Dict[str, Handler],
    env: Dict[str, Any],
    collect: Optional[List[Any]],
) -> None:
    for step in steps:
        op = step["op"]
        if op == "call":
            if step.get("interface") not in handlers:
                raise WorkflowError(f"unknown interface: {step.get('interface')}")
            in_ = _resolve(step.get("in", {}), env)
            env[step["into"]] = handlers[step["interface"]](in_)
        elif op == "assign":
            env[step["into"]] = _resolve(step["value"], env)
        elif op == "map":
            items = _resolve(step["over"], env)
            if not isinstance(items, list):
                raise WorkflowError(f"map over 不是列表: {step['over']}")
            as_var = step.get("as", "it")
            collected: List[Any] = []
            for el in items:
                env[as_var] = el
                _run_steps(step.get("do", []), handlers, env, collected)
            env[step["into"]] = collected
        elif op == "emit":
            if collect is None:
                raise WorkflowError("emit 只能在 map 内部使用")
            collect.append(_resolve(step["value"], env))
        elif op == "if":
            cond = mcel.evaluate(step["when"], env)
            branch = step["then"] if cond else step.get("else", [])
            _run_steps(branch, handlers, env, collect)
        else:
            raise WorkflowError(f"unknown op: {op}")
