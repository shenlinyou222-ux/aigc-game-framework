# -*- coding: utf-8 -*-
"""LLM 注入渲染: 无损 JSON compact(2026-09-04 实测决策)。

- 无损: 除 semantic_vector 外全部字段保留(向量不入 prompt——它的价值在检索层
  已兑现, 1024 维文本形态对 LLM 是 ~5000 token 的纯噪声)。
- 预算: 单条 ~100 token 上下, top-8 注入 ~600 token; 调用方可用 budget 截断。
"""

from __future__ import annotations

import json
from typing import Iterable


def render_context(records: Iterable, *, include_session: bool = False) -> str:
    """把召回记录渲染为 JSON compact 注入文本(无损, 向量除外)。

    接受 GameLogRecord 或 HybridRecallResult(自动解包 .record)。"""
    items = []
    for item_ in records:
        r = getattr(item_, "record", item_)
        item = {
            "event_id": r.event_id,
            "event_type": r.event_type,
            "level": r.level,
            "time": r.time,
            "unix_time": r.unix_time,
            "frame": r.frame,
            "data": r.data,
            "tags": r.tags,
        }
        if include_session:
            item["session_id"] = r.session_id
        items.append(item)
    return json.dumps(items, ensure_ascii=False, separators=(",", ":"))


def estimate_tokens(text: str, chars_per_token: float = 3.0) -> int:
    """粗估 token 数(中文 ~1.5 字/token, 英文 ~4 字符/token, 混合取 3)。"""
    return max(1, int(len(text) / chars_per_token))
