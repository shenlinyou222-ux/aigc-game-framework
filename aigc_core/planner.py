"""planner —— 每轮“程序发起预组装 + LLM 结构化取舍”的协议层。

见 ../aigc_docs/06-会话上下文检索与工具组装机制.md 第 4 节：

1. 程序组 planner 输入（很小）：当前 user 请求（轻量清洗后的 raw 句）
   + 当前 WorkingSet 清单（只 id/kind/chars）+ 最近 1~2 轮摘要指针 + 预算。
2. 一次结构化 LLM 调用，只允许输出 JSON：
   {
     "intent": "recall | none",
     "scope":  ["fact", "event", "tool"],
     "queries": ["<raw 子句1>", ...],
     "load":  ["fact/xx", ...],
     "drop":  ["fact/xx", ...]
   }
3. 程序执行：queries 原样召回；校验 load/drop 预算 → 更新 manifest；
   命中/暗示工具 → 一并从工具向量库取 spec 组装。
4. 程序把 decision.messages 里“整段历史重放段”替换为
   “WorkingSet 展开 + 本轮 user 请求（+ 需要的工具 spec）”。

本模块不调用 LLM，只负责：输入组包、JSON 解析/校验、消息替换。
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field, field_validator

SCOPES = ("fact", "event", "tool")


class PlanResult(BaseModel):
    """LLM planner 的结构化输出（只允许这些键）。"""

    intent: Literal["recall", "none"] = "recall"
    scope: List[str] = Field(default_factory=lambda: ["fact", "event", "tool"])
    queries: List[str] = Field(default_factory=list)   # 轻量清洗后的 raw 子句，不改写
    load: List[str] = Field(default_factory=list)      # 候选 id（可带 kind 前缀由程序解析）
    drop: List[str] = Field(default_factory=list)

    @field_validator("scope")
    @classmethod
    def _scope_allowed(cls, v: List[str]) -> List[str]:
        bad = [s for s in v if s not in SCOPES]
        if bad:
            raise ValueError(f"scope contains unknown layer(s): {bad}")
        return list(v)

    @field_validator("queries", "load", "drop")
    @classmethod
    def _strings_only(cls, v: List[str]) -> List[str]:
        return [str(x) for x in v]


def build_planner_input(
    user_request: str,
    working_set_snapshot: Optional[Dict[str, Any]] = None,
    recent_pointers: Optional[List[str]] = None,
    query_clauses: Optional[List[str]] = None,
    aliases: Optional[Dict[str, str]] = None,
) -> Dict[str, Any]:
    """组很小、给 LLM planner 的输入（只 id/kind/chars + raw 请求 + 预算）。"""
    from .query_normalize import QueryNormalizer

    clauses = query_clauses or QueryNormalizer(aliases=aliases).normalize(user_request)
    return {
        "current_request_raw": user_request,
        "queries": clauses,
        "working_set": working_set_snapshot or {},
        "recent_session_pointers": list(recent_pointers or []),
        "budget": {
            "maxCount": (working_set_snapshot or {}).get("maxCount", 24),
            "maxTokens": (working_set_snapshot or {}).get("maxTokens", 16000),
        },
        "note": "只输出结构化 JSON，不写自由长文本；queries 为轻清洗后 raw 子句，勿改写语义",
    }


def _strip_fence(text: str) -> str:
    """去掉 LLM 可能加的 markdown 围栏（三个反引号 json ... 三个反引号）。"""
    fence = chr(96) * 3            # 三个反引号
    if not text.startswith(fence):
        return text
    lines = text.splitlines()
    lines = lines[1:]              # 去掉首行围栏
    if lines and lines[-1].strip() == fence:
        lines = lines[:-1]
    cleaned = "\n".join(lines).strip()
    if cleaned.startswith("json"):
        cleaned = cleaned[4:].strip()
    return cleaned


def parse_plan(raw: str) -> PlanResult:
    """把 LLM 返回的 JSON 文本解析成 PlanResult；容错去掉围栏/前后缀。"""
    text = _strip_fence((raw or "").strip())
    try:
        data = json.loads(text)
    except Exception as exc:
        raise ValueError(f"planner output is not valid JSON: {exc}") from exc
    if not isinstance(data, dict):
        raise ValueError("planner output must be a JSON object")
    # 只接受已知键：防 LLM 塞自由长文本/别的东西
    allowed = {"intent", "scope", "queries", "load", "drop"}
    extra = set(data) - allowed
    if extra:
        raise ValueError(f"planner output has unexpected keys: {sorted(extra)}")
    plan = PlanResult.model_validate(data)
    if plan.intent == "none":
        plan.queries = []
        plan.load = []
    return plan


def split_id(item_id: str) -> Dict[str, str]:
    """把 "fact/a1" / "event/s1" / "tool/play_music" 拆成 {kind, id}。

    无前缀时回退 kind="event"（向后兼容候选只有裸 id 的场景）。
    """
    if "/" in item_id:
        kind, _, _rest = item_id.partition("/")
        if kind in SCOPES:
            return {"kind": kind, "id": item_id}
    return {"kind": "event", "id": item_id}


HISTORY_REPLAY_PLACEHOLDER = "[HISTORY_REPLAY]"


def assemble_messages(
    working_set_expansion: List[Dict[str, Any]],
    user_request: str,
    tool_specs: Optional[List[Dict[str, Any]]] = None,
    history_replay: str = HISTORY_REPLAY_PLACEHOLDER,
) -> List[Dict[str, Any]]:
    """消息替换：把“整段历史重放段”换成 “WorkingSet 展开 + 工具 spec + 本轮 user 请求”。

    这是 DSH messages 替换的关键动作：不再重放整段历史，只放被展开的 id 正文。
    """
    working_block = {
        "role": "context",
        "type": "working_set",
        "content": [
            {
                "id": item["id"],
                "kind": item["kind"],
                "text": item.get("text", ""),
            }
            for item in working_set_expansion
        ],
    }
    tools_block: List[Dict[str, Any]] = []
    for spec in tool_specs or []:
        tools_block.append({"role": "context", "type": "tool_spec", "content": spec})
    user_block = {"role": "user", "content": user_request}
    return [
        {"role": "context", "type": "history_note", "content": history_replay},
        working_block,
        *tools_block,
        user_block,
    ]
