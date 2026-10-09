"""python_runner.py —— 受控执行器（极简版）。

LLM 用内部函数写 Python 主函数，这里负责受控执行，只做两件事：
1. 简单语法/调用校验（AST 白名单）：
   - 语法错误 -> 拒绝
   - 禁止 import / 属性访问(如 .system/.open) / 调未注册的“非安全”函数
   - 只允许：内部函数通过 `run("name", params)` 调用 + 少量安全内建
2. fallback 逻辑：
   - 语法错 / 命中 CEL 规则(RuleError) / 运行期错 -> 走 on_error fallback
   - 不 fallback 就原样抛（方便上层统一收 RuleError）

约定：body 里必须定义一个 `main(ctx)`，runner 用 env 作为 ctx 调用它；
内部接口用 `run("函数名", {params})`（可带第三个 env_overrides）。
"""

from __future__ import annotations

import ast
import builtins
from typing import Any, Callable, Dict, Optional

from .functions import FunctionRegistry, RuleError

_SAFE_BUILTINS = {
    "len", "range", "enumerate", "zip", "reversed", "sorted", "min", "max",
    "sum", "any", "all", "abs", "round", "str", "int", "float", "bool",
    "list", "dict", "tuple", "set", "isinstance", "isinstance", "repr",
}

_ALLOWED_CALLS = _SAFE_BUILTINS | {"run"}

ErrorHandler = Callable[[BaseException], Any]


class PythonRunnerError(RuntimeError):
    pass


def validate_body(body: str) -> ast.Module:
    """简单语法 + 白名单校验。非法 -> PythonRunnerError。"""
    try:
        tree = ast.parse(body, mode="exec")
    except SyntaxError as exc:
        raise PythonRunnerError(f"syntax error: {exc.msg} at line {exc.lineno}") from exc
    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            raise PythonRunnerError("import is not allowed")
        if isinstance(node, ast.Attribute):
            # 禁止 a.b（可规避 os.system / __class__ 等）；用下标 d["k"] 访问数据
            raise PythonRunnerError("attribute access is not allowed; use d['k']")
        if isinstance(node, ast.Call):
            func = node.func
            if isinstance(func, ast.Name) and func.id in _ALLOWED_CALLS:
                continue
            if isinstance(func, ast.Name):
                raise PythonRunnerError(f"calling unknown function: {func.id}")
            raise PythonRunnerError("only direct function calls allowed (run / safe builtins)")
    return tree


def run_python(
    body: str,
    registry: FunctionRegistry,
    env: Optional[Dict[str, Any]] = None,
    on_error: Optional[ErrorHandler] = None,
) -> Any:
    """受控执行 LLM 写的 Python 主函数。

    必须定义 main(ctx)。成功 -> main 的返回值；
    出错(语法/规则/运行期) -> 若有 on_error 则调用它并返回其结果，否则抛。
    """
    env = dict(env or {})
    safe_globals = {
        name: getattr(builtins, name) for name in _SAFE_BUILTINS if hasattr(builtins, name)
    }
    safe_globals["run"] = _make_run(registry, env)
    safe_globals["ctx"] = env

    try:
        tree = validate_body(body)
        code = compile(tree, "<llm-main>", "exec")
        ns: Dict[str, Any] = dict(safe_globals)
        exec(code, ns)  # noqa: S102 - 白名单已限制
        main = ns.get("main")
        if main is None or not callable(main):
            raise PythonRunnerError("body must define a callable main(ctx)")
        return main(env)
    except (PythonRunnerError, RuleError, Exception) as exc:  # noqa: BLE001
        if on_error is not None:
            return on_error(exc)
        raise


def _make_run(registry: FunctionRegistry, env: Dict[str, Any]) -> Callable[..., Any]:
    def run(name: str, params: Dict[str, Any], overrides: Optional[Dict[str, Any]] = None) -> Any:
        merged = dict(env)
        if overrides:
            merged.update(overrides)
        return registry.call(name, params, merged)

    return run
