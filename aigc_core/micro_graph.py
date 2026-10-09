"""micro-graph：参考 graphrunner 精简出的“内部函数即节点”Python 编排内核。

理念（对齐前文讨论）：
- 内部接口即节点：注册一张 name -> fn(ctx?, in)->out 的表，dispatch 即执行。
- 显式输入/输出绑定：GraphSpec{Nodes,Bindings,Input}，边用 JSON 路径(含 [N])。
- 校验退化 = 检索(resolvable) + 关系判断(relation)：预检阶段先把每个节点的入参
  从 上游输出/顶层输入 里“找出来绑定好”，再用 mini-CEL 规则(mcel.py)做关系判断。
- 执行同步；哪步错哪步抛(NodeError)，不续跑，但保留 step 日志上下文。
- 预留 async：给的是 arun()，handler 可返回 awaitable。
"""

from __future__ import annotations

import asyncio
import inspect
import re
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

from . import mcel

# handler: 内部函数即节点。in/out 都是 JSON 对象(map)。
Handler = Callable[[Dict[str, Any]], Dict[str, Any]]
AsyncHandler = Callable[[Dict[str, Any]], Any]  # 返回值可为 awaitable


@dataclass
class NodeSpec:
    key: str
    name: str
    # 图展开声明（graph-json 形式）：expand = {
    #   "over": <上游节点 key，其输出是列表>,
    #   "field": <上游输出里列表的 JSON 路径，如 "Data.items">,
    #   "as": <把每个元素绑到本节点哪个入参字段, 默认 "item">,
    #   "key_field": <可选：从元素取某字段当展开节点 key 后缀>,
    # }
    expand: Optional[Dict[str, Any]] = None


@dataclass
class Binding:
    from_node: str
    from_field: str
    to_node: str
    to_field: str
    transform: Optional[str] = None  # "first" 或 "at:N"(取第 N 项)；越界由 optional 决定
    optional: bool = False  # True：源缺失/越界/空 → 该节点输出空占位，不报错


@dataclass
class GraphSpec:
    nodes: List[NodeSpec]
    bindings: List[Binding]
    input: Dict[str, Any] = field(default_factory=dict)


def graph_from_dict(data: Dict[str, Any]) -> GraphSpec:
    """从图 JSON dict 构造 GraphSpec（供 “以图 json 形式表达” 使用）。"""
    return GraphSpec(
        nodes=[NodeSpec(**n) for n in data.get("nodes", [])],
        bindings=[Binding(**b) for b in data.get("bindings", [])],
        input=dict(data.get("input", {}) or {}),
    )


def graph_to_dict(graph: GraphSpec) -> Dict[str, Any]:
    """把 GraphSpec 转回可序列化的图 JSON dict。"""
    return {
        "nodes": [
            {k: v for k, v in n.__dict__.items() if v is not None}
            for n in graph.nodes
        ],
        "bindings": [
            {k: v for k, v in b.__dict__.items() if v is not None}
            for b in graph.bindings
        ],
        "input": graph.input,
    }


@dataclass
class StepLog:
    key: str
    name: str
    status: str  # success / failed
    input: Dict[str, Any] = field(default_factory=dict)
    output: Any = None
    error: Optional[Exception] = None


@dataclass
class RunResult:
    ok: bool
    outputs: Dict[str, Any] = field(default_factory=dict)
    steps: List[StepLog] = field(default_factory=list)
    failed_step: Optional[str] = None
    error: Optional[Exception] = None


class NodeError(RuntimeError):
    def __init__(self, node_key: str, name: str, cause: Exception, log: List[StepLog]) -> None:
        super().__init__(f"node {node_key}({name}) failed: {cause}")
        self.node_key = node_key
        self.name = name
        self.cause = cause
        self.log = log


# ---------------------------------------------------------------------------
# JSON 路径取值（上游输出 / 顶层输入 -> 具体字段），对应 graphrunner NavigatePath
# ---------------------------------------------------------------------------

_PATH_SPLIT = re.compile(r"[.\[\]]+")


def navigate(data: Any, path: str) -> Any:
    """按 JSON 路径取值。例：navigate(out, "Data.ClusterSet[0].ClusterId")。"""
    if not path:
        return data
    cur = data
    for seg in [s for s in _PATH_SPLIT.split(path) if s != ""]:
        if isinstance(cur, dict):
            if seg not in cur:
                raise KeyError(f"micro_graph: no field {seg!r} in map (path={path!r})")
            cur = cur[seg]
        elif isinstance(cur, list):
            idx = int(seg)
            try:
                cur = cur[idx]
            except IndexError:
                raise IndexError(
                    f"micro_graph: index {idx} out of range (path={path!r})"
                ) from None
        else:
            raise KeyError(
                f"micro_graph: cannot descend into {type(cur).__name__} (path={path!r})"
            )
    return cur


def _apply_transform(value: Any, transform: Optional[str]) -> Any:
    if transform is None:
        return value
    if transform == "first":
        if isinstance(value, list) and value:
            return value[0]
        return value
    m = re.fullmatch(r"at:(-?\d+)", transform)
    if m:
        idx = int(m.group(1))
        if isinstance(value, list):
            try:
                return value[idx]
            except IndexError:
                raise IndexError(
                    f"micro_graph: transform {transform!r} index out of range (len={len(value)})"
                ) from None
        raise TypeError(f"micro_graph: transform {transform!r} needs a list, got {type(value).__name__}")
    raise ValueError(f"micro_graph: unknown transform {transform!r}")


# ---------------------------------------------------------------------------
# 绑定：把上游/顶层输入按边塞进某个节点的入参
# ---------------------------------------------------------------------------


def bind_node_input(
    node: NodeSpec,
    graph: GraphSpec,
    outputs: Dict[str, Any],
    top_input: Dict[str, Any],
) -> Dict[str, Any]:
    """显式绑定：只取 to_node == node.key 的边。from_node == 'START' 读顶层输入。"""
    resolved: Dict[str, Any] = {}
    for b in graph.bindings:
        if b.to_node != node.key:
            continue
        try:
            if b.from_node == "START":
                src = top_input
            else:
                if b.from_node not in outputs:
                    raise KeyError(
                        f"micro_graph: node {node.key} 依赖 {b.from_node}，但上游未产生输出"
                    )
                src = outputs[b.from_node]
            resolved[b.to_field] = _apply_transform(navigate(src, b.from_field), b.transform)
        except Exception:
            if b.optional:
                # optional 边取不到 → 该字段缺席（不填），不报错
                continue
            raise
    return resolved


def topo_order(graph: GraphSpec) -> List[NodeSpec]:
    """按边(除 START)做拓扑排序；有环报错。"""
    idx = {n.key: i for i, n in enumerate(graph.nodes)}
    indeg: Dict[str, int] = {n.key: 0 for n in graph.nodes}
    edges: Dict[str, List[str]] = {n.key: [] for n in graph.nodes}
    for b in graph.bindings:
        if b.from_node in ("START", "__start__"):
            continue
        if b.from_node not in idx or b.to_node not in idx:
            continue
        edges[b.from_node].append(b.to_node)
        indeg[b.to_node] += 1
    order: List[NodeSpec] = []
    queue = [n for n in graph.nodes if indeg[n.key] == 0]
    while queue:
        n = queue.pop(0)
        order.append(n)
        for m in edges[n.key]:
            indeg[m] -= 1
            if indeg[m] == 0:
                queue.append(next(x for x in graph.nodes if x.key == m))
    if len(order) != len(graph.nodes):
        raise ValueError("micro_graph: graph has a cycle")
    return order


# ---------------------------------------------------------------------------
# 图展开（graph-json 表达：把“按列表 fan-out”的模板节点 unroll 成 N 个节点）
# ---------------------------------------------------------------------------


def expand_graph(
    graph: GraphSpec,
    source_outputs: Dict[str, Any],
) -> GraphSpec:
    """把图里带 `expand` 的模板节点展开成具体的 N 个节点 + N 条索引边。

    - 模板节点：`NodeSpec(key, name, expand={"over","field","as","key_field"})`
    - source_outputs：上游(expand.over)已产出的结果（展开器据此知道列表长度）
    - 空列表 → 模板节点被移除（不产生任何具体节点），即“返回空、不调接口”
    - 模板节点的“常量边”（非 expand 来源的其它 binding）会复制到每个展开节点
    """
    template_keys = {n.key for n in graph.nodes if n.expand}
    out_nodes: List[NodeSpec] = []
    out_bindings: List[Binding] = []

    for n in graph.nodes:
        if not n.expand:
            out_nodes.append(n)
            continue
        exp = n.expand
        over = exp["over"]
        field = exp["field"]
        as_field = exp.get("as", "item")
        key_field = exp.get("key_field")

        if over not in source_outputs:
            raise KeyError(f"micro_graph: expand over {over} 无输出")
        items = navigate(source_outputs[over], field)
        if not isinstance(items, list):
            raise TypeError(f"micro_graph: expand field {field!r} 不是列表(got {type(items).__name__})")

        # 常量边：原本指向模板节点、且 not from over 的边（例如从 START 给 token）
        consts = [
            b
            for b in graph.bindings
            if b.to_node == n.key and b.from_node != over
        ]
        for i, el in enumerate(items):
            suffix = str(i)
            if key_field and isinstance(el, dict) and key_field in el:
                suffix = str(el[key_field])
            key = f"{n.key}_{suffix}"
            out_nodes.append(NodeSpec(key=key, name=n.name))
            # 每个元素 = 一条索引边：over.field[i] -> key.as_field
            out_bindings.append(
                Binding(from_node=over, from_field=f"{field}[{i}]", to_node=key, to_field=as_field)
            )
            for c in consts:
                out_bindings.append(
                    Binding(from_node=c.from_node, from_field=c.from_field,
                            to_node=key, to_field=c.to_field, transform=c.transform, optional=c.optional)
                )

    # 保留所有“不指向模板节点”的边
    for b in graph.bindings:
        if b.to_node not in template_keys:
            out_bindings.append(b)

    return GraphSpec(nodes=out_nodes, bindings=out_bindings, input=graph.input)


# ---------------------------------------------------------------------------
# 校验：检索 + 关系判断（mini-CEL）
# ---------------------------------------------------------------------------


def validate_node_input(node: NodeSpec, input_: Dict[str, Any], rules: Optional[Dict[str, str]]) -> None:
    """rules: node.key -> mini-CEL 表达式。表达式的变量即该节点解析好的入参。"""
    if not rules:
        return
    expr = rules.get(node.key)
    if not expr:
        return
    ok = mcel.evaluate(expr, input_)
    if ok is not True:
        raise mcel.McelError(f"micro_graph: rule failed for node {node.key}: {expr}")


# ---------------------------------------------------------------------------
# 预检：只“找参数+绑定+校验”，不执行 handler（两段式：先绑死再开跑）
# ---------------------------------------------------------------------------


def preflight(
    graph: GraphSpec,
    rules: Optional[Dict[str, str]] = None,
    input_override: Optional[Dict[str, Any]] = None,
    output_templates: Optional[Dict[str, Dict[str, Any]]] = None,
) -> Dict[str, Dict[str, Any]]:
    """同步预检：不执行 handler，只为每个节点“找参数+绑定+关系校验”。

    - 绑定只读顶层输入(START)与上游**输出样例 schema**（output_templates，
      对应 graphrunner 的 output_schema，给出节点输出形状即可静态下钻）。
    - 校验用 rules(mini-CEL) 做关系判断。
    - 任一节点解析不到 / 规则不成立 → 直接抛错，不进入执行阶段。
    返回 node.key -> 已绑定好的入参（可直接喂给执行）。
    """
    top_input: Dict[str, Any] = dict(graph.input)
    if input_override:
        top_input.update(input_override)
    templates = dict(output_templates or {})

    outputs: Dict[str, Any] = {}
    bound: Dict[str, Dict[str, Any]] = {}
    for node in topo_order(graph):
        in_ = bind_node_input(node, graph, outputs, top_input)
        validate_node_input(node, in_, rules)
        bound[node.key] = in_
        # 预检不真跑 handler，下游解析用“输出样例”代替真实输出。
        outputs[node.key] = templates.get(node.key, {})
    return bound


# ---------------------------------------------------------------------------
# 执行
# ---------------------------------------------------------------------------


def _merge_input(graph_input: Dict[str, Any], input_override: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    merged = dict(graph_input)
    if input_override:
        merged.update(input_override)
    return merged


def run(
    graph: GraphSpec,
    handlers: Dict[str, Handler],
    rules: Optional[Dict[str, str]] = None,
    input_override: Optional[Dict[str, Any]] = None,
) -> RunResult:
    """同步执行整张图。哪步错在哪步抛 NodeError（已带 step 日志上下文）。"""
    top_input = _merge_input(graph.input, input_override)
    outputs: Dict[str, Any] = {}
    steps: List[StepLog] = []
    for node in topo_order(graph):
        try:
            in_ = bind_node_input(node, graph, outputs, top_input)
            validate_node_input(node, in_, rules)
            out = handlers[node.name](in_)
            if not isinstance(out, dict):
                raise TypeError(
                    f"micro_graph: handler {node.name} must return dict, got {type(out).__name__}"
                )
        except Exception as exc:  # noqa: BLE001 - 任意一步失败就地抛
            steps.append(StepLog(node.key, node.name, "failed", input=in_ if "in_" in locals() else {}, error=exc))
            raise NodeError(node.key, node.name, exc, steps) from exc
        outputs[node.key] = out
        steps.append(StepLog(node.key, node.name, "success", input=in_, output=out))
    return RunResult(ok=True, outputs=outputs, steps=steps)


async def arun(
    graph: GraphSpec,
    handlers: Dict[str, AsyncHandler],
    rules: Optional[Dict[str, str]] = None,
    input_override: Optional[Dict[str, Any]] = None,
) -> RunResult:
    """异步执行（handler 可 async）。错误语义同 run()：就地抛、留日志。"""
    top_input = _merge_input(graph.input, input_override)
    outputs: Dict[str, Any] = {}
    steps: List[StepLog] = []
    for node in topo_order(graph):
        try:
            in_ = bind_node_input(node, graph, outputs, top_input)
            validate_node_input(node, in_, rules)
            out = handlers[node.name](in_)
            if inspect.isawaitable(out):
                out = await out
            if not isinstance(out, dict):
                raise TypeError(
                    f"micro_graph: handler {node.name} must return dict, got {type(out).__name__}"
                )
        except Exception as exc:  # noqa: BLE001
            steps.append(StepLog(node.key, node.name, "failed", input=in_ if "in_" in locals() else {}, error=exc))
            raise NodeError(node.key, node.name, exc, steps) from exc
        outputs[node.key] = out
        steps.append(StepLog(node.key, node.name, "success", input=in_, output=out))
    return RunResult(ok=True, outputs=outputs, steps=steps)
