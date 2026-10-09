"""fact_store —— 事实层（FactStore）。

把“脏过程”写压缩成干净长期事实。与 event/tool 走同一套 LogVectorStore 记录形态
（复用 GameLogRecord + kind="fact"），因此：
- 与事件同构、可被同一套召回（scope: fact）；
- 长期：session_id 留空（全局），不依赖短会话死期；
- 写压缩 = 相同文本去重，避免把同一事实反复堆成噪声。

见 ../aigc_docs/06-会话上下文检索与工具组装机制.md 第 5/6 节。
"""

from __future__ import annotations

import time
import uuid
from typing import Any, Dict, List, Optional

from .log_vector_store import LogVectorStore
from .models import GameLogRecord, HybridRecallResult

FACT_KIND = "fact"
FACT_EVENT_TYPE = "fact"


class FactStore:
    """把干净事实写进共享 LogVectorStore（kind="fact"）并提供事实层检索。"""

    def __init__(
        self,
        log_store: LogVectorStore,
        event_id_prefix: str = "fact/",
    ) -> None:
        self._store = log_store
        self._prefix = event_id_prefix

    # ---------- 写压缩 / 入库 ----------

    def ingest(
        self,
        text: str,
        event_id: Optional[str] = None,
        tags: Optional[List[str]] = None,
        source_type: str = "",
    ) -> GameLogRecord:
        """写入一条干净事实。相同 text 已存在则返回既有条目（写压缩去重）。"""
        text = (text or "").strip()
        if not text:
            raise ValueError("fact text must not be empty")
        # 写压缩：命中同文本事实直接返回，不重复堆库
        existing = self._find_by_text(text)
        if existing is not None:
            return existing

        now = time.time()
        record = GameLogRecord(
            event_id=event_id or (self._prefix + uuid.uuid4().hex[:10]),
            event_type=FACT_EVENT_TYPE,
            kind=FACT_KIND,
            unix_time=now,
            last_used_at=now,
            session_id="",           # 全局长期
            data={"text": text, "tags": list(tags or []), "source_type": source_type},
        )
        self._store.add(record)
        return record

    def search(
        self,
        text: str,
        limit: int = 5,
        session_id: Optional[str] = None,
    ) -> List[HybridRecallResult]:
        """事实层混合召回（dense 可空：靠词法/时效/同层即可召回）。"""
        return self._store.recall_hybrid(
            query_text=text,
            session_id=session_id,
            event_type=FACT_EVENT_TYPE,
            limit=limit,
            lexical_weight=1.0,
            recency_weight=0.0,   # 事实层长期：不看近因，看是否命中
        )

    def get(self, event_id: str) -> Optional[GameLogRecord]:
        record = self._store.get(event_id)
        if record is None or (record.kind or "") != FACT_KIND:
            return None
        return record

    def count(self) -> int:
        return sum(1 for r in self._store.all() if (r.kind or "") == FACT_KIND)

    def all(self) -> List[GameLogRecord]:
        return [r for r in self._store.all() if (r.kind or "") == FACT_KIND]

    # ---------- internal ----------

    def _find_by_text(self, text: str) -> Optional[GameLogRecord]:
        for record in self._store.all():
            if (record.kind or "") != FACT_KIND:
                continue
            if record.data.get("text") == text:
                return record
        return None
