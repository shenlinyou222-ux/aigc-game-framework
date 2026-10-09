"""内存版公共向量知识库。

当前先实现结构化过滤和生命周期/写入权限控制；
后续替换为 FAISS/ChromaDB + embedding 时保持 MemoryStore 接口不变。
"""

from __future__ import annotations

import threading
from typing import Dict, Iterable, List, Optional

from .models import Lifecycle, MemoryEntry, MemoryQuery, WriterRole


class MemoryStoreError(Exception):
    pass


class InMemoryMemoryStore:
    """线程安全的内存记忆库。"""

    def __init__(self) -> None:
        self._entries: Dict[str, MemoryEntry] = {}
        self._lock = threading.RLock()

    def add(self, entry: MemoryEntry, writer: WriterRole) -> MemoryEntry:
        """写入前做生命周期权限校验。

        - S 级：只有程序（PROGRAM）可写。
        - A 级：程序可写；模型如需写，应由主控判定是否允许，这里先禁止模型直接写 A。
        - B/C 级：程序、模型、感知组件都可以写。
        """
        if writer != WriterRole.PROGRAM:
            if entry.lifecycle == Lifecycle.S:
                raise MemoryStoreError("S 级永久记忆仅允许程序写入")
            if entry.lifecycle == Lifecycle.A and writer == WriterRole.MODEL:
                raise MemoryStoreError("A 级长期记忆不允许模型直接写入，需程序落库")

        with self._lock:
            self._entries[entry.event_id] = entry.model_copy(deep=True)
            return entry

    def get(self, event_id: str) -> Optional[MemoryEntry]:
        with self._lock:
            entry = self._entries.get(event_id)
            return entry.model_copy(deep=True) if entry else None

    def delete(self, event_id: str) -> bool:
        """删除仅由主控调用；对外不直接暴露。"""
        with self._lock:
            return self._entries.pop(event_id, None) is not None

    def query(self, query: MemoryQuery) -> List[MemoryEntry]:
        with self._lock:
            candidates = list(self._entries.values())

        def match(entry: MemoryEntry) -> bool:
            if query.event_type and entry.exact_fields.type != query.event_type:
                return False
            if query.entity_id and entry.exact_fields.entity_id != query.entity_id:
                return False
            if query.location and entry.location != query.location:
                return False
            if query.source_faction and entry.source_faction != query.source_faction:
                return False
            if query.lifecycle and entry.lifecycle != query.lifecycle:
                return False
            if entry.confidence < query.min_confidence:
                return False
            if query.since is not None and entry.timestamp < query.since:
                return False
            if query.until is not None and entry.timestamp > query.until:
                return False
            return True

        matched = [e for e in candidates if match(e)]
        matched.sort(key=lambda e: e.timestamp, reverse=True)
        return [e.model_copy(deep=True) for e in matched[: query.limit]]

    def all(self) -> Iterable[MemoryEntry]:
        with self._lock:
            return [e.model_copy(deep=True) for e in self._entries.values()]
