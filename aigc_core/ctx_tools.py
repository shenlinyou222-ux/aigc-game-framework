"""ctx_tools —— 留给 LLM 主动查向量的口子（补充通道，不依赖它）。

见 ../aigc_docs/06-会话上下文检索与工具组装机制.md 第 4.2 节：
- search_context(query, scope, k)  -> 原样进后端，返回候选；LLM 可二次 load。
- load_context(id...)              -> 改 WorkingSet manifest。
- drop_context(id...)              -> 改 WorkingSet manifest。

这套工具和预组装用同一后端、同一 manifest，只是发起方从程序换成 LLM。
用 functions.py 的 InternalFunction 包装，可放进受控执行器/编排，走同一套规则与调用协议。
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from .functions import FunctionRegistry, InternalFunction


def register_context_tools(
    registry: FunctionRegistry,
    controller: Any,
    session: str = "",
) -> FunctionRegistry:
    """把三个“主动查上下文”口子注册进 FunctionRegistry（env 无关，闭包绑定 controller）。"""

    def _search(params: Dict[str, Any]) -> Dict[str, Any]:
        query = str(params.get("query", ""))
        scope = params.get("scope") or ["fact", "event", "tool"]
        k = int(params.get("k", params.get("top_k", 5)))
        return {"candidates": controller.recall_candidates(query, session=session, scope=scope, limit=k)}

    def _load(params: Dict[str, Any]) -> Dict[str, Any]:
        raw = params.get("ids", params.get("id", []))
        if isinstance(raw, str):
            raw = [raw]
        ids: List[Dict[str, str]] = []
        for item in raw:
            if isinstance(item, dict):
                ids.append({"id": str(item["id"]), "kind": str(item.get("kind", ""))})
            else:
                # 裸 id：让 controller 按前缀推断 kind
                ids.append({"id": str(item), "kind": ""})
        return controller.working_set_load(session, ids)

    def _drop(params: Dict[str, Any]) -> Dict[str, Any]:
        raw = params.get("ids", params.get("id", []))
        if isinstance(raw, str):
            raw = [raw]
        return controller.working_set_drop(session, [str(x) for x in raw])

    registry.register(
        InternalFunction(
            name="search_context",
            description="主动查向量上下文：返回 [id, kind, score, snippet] 候选，供后续 load_context 选择",
            fn=_search,
            input_fields={"query": "自然语言查询", "scope": "查询层数组: fact/event/tool", "k": "返回条数"},
        )
    ).register(
        InternalFunction(
            name="load_context",
            description="把候选 id 载入 WorkingSet（先过预算再拉正文展开）",
            fn=_load,
            input_fields={"ids": "id 数组或单个 id"},
        )
    ).register(
        InternalFunction(
            name="drop_context",
            description="把 id 从 WorkingSet 卸载（只改 manifest）",
            fn=_drop,
            input_fields={"ids": "id 数组或单个 id"},
        )
    )
    return registry
