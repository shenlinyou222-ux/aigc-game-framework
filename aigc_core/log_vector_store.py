"""日志向量库（JSONL 离线 demo 层）。

⚠️ 这不是“向量数据库”，只是离线/未装库时的临时代替（内存 dict + JSONL）。
真正的向量库（ChromaDB / LanceDB / Qdrant / Milvus…）见 `vector_db.py`：
**数据持久化与到期清理应由数据库负责**，本类的手动生命周期只用于旁路调试。

功能（demo 用）：
- 向量写入/召回（纯 Python 余弦，后续由真库替换）
- event_type / session / 时间窗口等结构化过滤
- JSONL 持久化，方便断点续读和旁路调试

设计上不替代 `memory.py` 的公共向量知识库：memory 保存“世界记忆/事实”，
这里保存“原始游戏日志流/DSH 上下文流”，供分析、回放、AI 上下文回溯使用。
"""

from __future__ import annotations

import json
import math
import os
import re
import threading

import numpy as np
import time
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Protocol

from .lifecycle import VectorLifecycleManager
from .entity_index import EntityIndex, entity_raw_hits
from .models import (
    GameLogRecord,
    HybridRecallResult,
    LogSearchResult,
    LogVectorQuery,
)


@dataclass
class StoreLifecycleConfig:
    """日志向量库的“何时删除”配置（按条数上限 + 空闲保留时间）。

    retention_sec: 记录距 last_used_at 空闲超过该秒数即被删除；<=0 表示不按时间删。
    max_entries:    硬上限（按条数）；<=0 表示不限条数。
    warn_ratio:     警戒线比例，超过 int(max_entries * warn_ratio) 条即算“接近满”，
                    可在写入/查询时返回状态或触发主动清理，避免撞顶。
    """

    retention_sec: float = 1800.0        # 默认空闲 30 分钟未用即删
    max_entries: int = 20000             # 默认最多 2 万条
    warn_ratio: float = 0.8              # 默认 80% 为警戒线

    @property
    def warn_entries(self) -> int:
        return int(self.max_entries * self.warn_ratio)

    def to_dict(self) -> Dict[str, object]:
        return asdict(self)


# 生命周期默认配置文件（相对框架根目录，可在文件里直接改默认值）
DEFAULT_LIFECYCLE_CONFIG_PATH = (
    Path(__file__).resolve().parent.parent / "vector_store_lifecycle.json"
)


def load_lifecycle_config(
    path: Optional[os.PathLike[str] | str] = None,
) -> StoreLifecycleConfig:
    """从 JSON 配置读取生命周期参数；文件缺失时返回内置默认值。"""
    p = Path(path).expanduser() if path else DEFAULT_LIFECYCLE_CONFIG_PATH
    if p.exists():
        try:
            with p.open("r", encoding="utf-8") as f:
                data = json.load(f)
            return StoreLifecycleConfig(**data)
        except Exception:
            # 配置损坏时回退到默认值，不让启动失败
            return StoreLifecycleConfig()
    return StoreLifecycleConfig()


def write_lifecycle_config(
    path: Optional[os.PathLike[str] | str] = None,
    config: Optional[StoreLifecycleConfig] = None,
) -> Path:
    """把生命周期配置写成一个 JSON 文件，方便人工调整默认值。"""
    p = Path(path).expanduser() if path else DEFAULT_LIFECYCLE_CONFIG_PATH
    config = config or StoreLifecycleConfig()
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(
        json.dumps(config.to_dict(), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return p


class LogEmbedder(Protocol):
    """把一个日志记录变成向量的函数/对象。"""

    def __call__(self, record: GameLogRecord) -> List[float]: ...


class LogVectorStore:
    """线程安全的轻量日志向量库。

    不依赖 ChromaDB/FAISS；向量检索用余弦相似度实现。数据量小、本地验证足够。
    后续若换用正式向量库，只需替换本类内部实现，模型和查询接口不变。
    """

    def __init__(
        self,
        path: Optional[os.PathLike[str] | str] = None,
        config: Optional[StoreLifecycleConfig] = None,
        config_path: Optional[os.PathLike[str] | str] = None,
        lifecycle: Optional[VectorLifecycleManager] = None,
    ) -> None:
        """lifecycle: 可选；传入时启用“死期+持续时间”的可热更换生命周期管理，
        否则维持原有 last_used_at + retention_sec 行为（向后兼容）。"""
        self._path = Path(path).expanduser() if path else None
        self._entries: Dict[str, GameLogRecord] = {}
        self._persisted_ids: set[str] = set()
        self._lifecycle = lifecycle
        self._lock = threading.RLock()
        if config is not None:
            self._config = config
        elif config_path is not None:
            p = Path(config_path).expanduser()
            if not p.exists():
                # 显式指定的配置不存在时，落一份默认文件方便编辑
                write_lifecycle_config(p)
            self._config = load_lifecycle_config(p)
        else:
            # 默认读框架根目录的配置文件（存在即生效）；不存在用内置默认
            self._config = load_lifecycle_config()
        if self._path and self._path.exists():
            self._load(self._path)

    @property
    def path(self) -> Optional[Path]:
        return self._path

    @property
    def config(self) -> StoreLifecycleConfig:
        return self._config

    def count(self) -> int:
        with self._lock:
            return len(self._entries)

    def add(self, record: GameLogRecord) -> GameLogRecord:
        """写入一条日志；如果 event_id 已存在则覆盖内存中的记录。

        持久化文件只对“新 event_id”追加，避免重复读取后写回重复行。
        写入后做一次懒维护：清掉已空闲超时的记录，并保证不超过条数上限。
        """
        now = time.time()
        with self._lock:
            stored = record.model_copy(deep=True)
            if stored.last_used_at <= 0:
                stored.last_used_at = now
            is_new = stored.event_id not in self._entries
            self._entries[stored.event_id] = stored
            if self._lifecycle is not None:
                if is_new:
                    # Create 生命周期策略（默认：duration=默认，deadline_base=now）
                    self._lifecycle.create(stored.event_id, now=now)
                else:
                    # 覆盖写入视为 Update：只刷新使用时间，不改 duration
                    self._lifecycle.touch(stored.event_id, now=now)
            if self._path and stored.event_id not in self._persisted_ids:
                self._path.parent.mkdir(parents=True, exist_ok=True)
                with self._path.open("a", encoding="utf-8") as f:
                    f.write(stored.model_dump_json() + "\n")
                self._persisted_ids.add(stored.event_id)
            self._sweep_expired_locked(now)
            self._evict_over_cap_locked(now)
            return record

    def add_many(self, records: Iterable[GameLogRecord]) -> int:
        count = 0
        for record in records:
            self.add(record)
            count += 1
        return count

    def get(self, event_id: str) -> Optional[GameLogRecord]:
        with self._lock:
            record = self._entries.get(event_id)
            return record.model_copy(deep=True) if record else None

    def delete(self, event_id: str) -> bool:
        """删除后若配置了持久化文件，会重写整个 JSONL 快照。"""
        with self._lock:
            existed = self._entries.pop(event_id, None) is not None
            if existed:
                if self._lifecycle is not None:
                    self._lifecycle.delete(event_id)
                self._persisted_ids.discard(event_id)
                if self._path:
                    self._save_locked(self._path)
            return existed

    def touch(self, event_id: str, at: Optional[float] = None) -> bool:
        """标记某条记录“被使用/被注入上下文”，刷新其空闲计时基准。

        这是“被用到就续命”的入口：调用方应在把该向量真正注入上下文之前
        调用它，且不覆盖原始 unix_time。只更新库内原条目（不是查询返回的副本）。
        """
        now = time.time() if at is None else float(at)
        with self._lock:
            record = self._entries.get(event_id)
            if record is None:
                return False
            record.last_used_at = now
            if self._lifecycle is not None:
                # 续命：重置死期基准 deadline_base=now，duration 保持不变
                self._lifecycle.touch(event_id, now=now)
            if self._path:
                self._save_locked(self._path)
            return True

    @property
    def lifecycle(self) -> Optional[VectorLifecycleManager]:
        """当前启用的生命周期管理器（死期+持续时间语义），未启用为 None。"""
        return self._lifecycle

    def lifecycle_meta(self, event_id: str) -> Optional[Dict[str, object]]:
        """取某向量生命周期元数据（deadline_base/duration/expires_at/state）。"""
        if self._lifecycle is None:
            return None
        meta = self._lifecycle.get(event_id)
        return meta.to_dict() if meta else None

    def lifecycle_expires_at(self, event_id: str) -> Optional[float]:
        """理论删除时刻 = deadline_base + duration_sec。"""
        if self._lifecycle is None:
            return None
        return self._lifecycle.expires_at(event_id)

    def set_lifecycle_duration(
        self,
        event_id: str,
        duration_sec: float,
    ) -> Optional[float]:
        """显式修改某向量的“持续时间”（一般不改；保留原死期基准）。

        返回新的理论删除时刻；未启用生命周期管理时返回 None。
        """
        if self._lifecycle is None:
            return None
        meta = self._lifecycle.set_duration(event_id, duration_sec)
        return meta.expires_at() if meta else None

    def sweep_expired(self, now: Optional[float] = None) -> int:
        """删除过期记录。

        - 未启用生命周期管理器：删除“距 last_used_at 空闲超过 retention_sec”的记录；
        - 启用后：按“死期+持续时间”(deadline_base+duration_sec <= now) 判定过期。
        """
        with self._lock:
            return self._sweep_expired_locked(time.time() if now is None else float(now))

    def evict_over_cap(self, now: Optional[float] = None) -> int:
        """当条数超过 max_entries 时，按“最久未被使用”淘汰到上限内。"""
        with self._lock:
            return self._evict_over_cap_locked(time.time() if now is None else float(now))

    def warning_breached(self) -> bool:
        """是否已越过警戒线（接近硬上限），可用来提前触发清理/预警。"""
        with self._lock:
            return self._config.max_entries > 0 and (
                len(self._entries) > self._config.warn_entries
            )

    def capacity_status(self) -> Dict[str, object]:
        """返回容量与删除策略的当前状态，便于观测/预警。"""
        with self._lock:
            return {
                "count": len(self._entries),
                "max_entries": self._config.max_entries,
                "warn_entries": self._config.warn_entries,
                "over_warning": self._config.max_entries > 0
                and len(self._entries) > self._config.warn_entries,
                "retention_sec": self._config.retention_sec,
            }

    def query(self, query: LogVectorQuery) -> List[LogSearchResult]:
        """查询日志。

        - query.query_vector 非空：对带向量的日志做余弦召回。
        - query.query_vector 为空：只做结构化过滤，最近日志优先。

        查询前会先懒删除已空闲超时的记录（生命周期管理）。
        """
        now = time.time()
        with self._lock:
            self._sweep_expired_locked(now)
            candidates_all = list(self._entries.values())

        def matches(record: GameLogRecord, tags: Optional[List[str]] = None) -> bool:
            if query.event_type and record.event_type != query.event_type:
                return False
            if not _session_matches(record, query.session_id):
                return False
            if query.since is not None and record.unix_time < query.since:
                return False
            if query.until is not None and record.unix_time > query.until:
                return False
            if tags:
                have = set(record.tags or ())
                if query.tag_match == "any":
                    if not any(t in have for t in tags):
                        return False
                else:
                    if not all(t in have for t in tags):
                        return False
            return True

        candidates = [r for r in candidates_all if matches(r, query.tags)]
        tag_matched = True
        if query.tags and not candidates:
            # 标签零命中：降级为无标签查询（与 SqliteLogStore 语义一致）
            candidates = [r for r in candidates_all if matches(r, None)]
            tag_matched = False

        if query.query_vector:
            results: List[LogSearchResult] = []
            for record in candidates:
                if not record.semantic_vector:
                    continue
                score = _cosine_similarity(query.query_vector, record.semantic_vector)
                if score is None or score < query.min_score:
                    continue
                results.append(
                    LogSearchResult(
                        record=record, score=round(score, 6), tag_matched=tag_matched
                    )
                )
            results.sort(key=lambda item: (item.score, item.record.unix_time), reverse=True)
            return results[: query.limit]

        candidates.sort(key=lambda record: record.unix_time, reverse=True)
        return [
            LogSearchResult(record=record.model_copy(deep=True), tag_matched=tag_matched)
            for record in candidates[: query.limit]
        ]

    def fetch_meta_candidates(
        self,
        event_type: Optional[str] = None,
        session_id: Optional[str] = None,
        since: Optional[float] = None,
        until: Optional[float] = None,
        tags: Optional[List[str]] = None,
        tag_match: str = "all",
    ) -> tuple[List[GameLogRecord], bool]:
        """按元数据(含标签门控/零命中降级)拉取候选记录, 供多库扇出复用。"""
        with self._lock:
            self._sweep_expired_locked(time.time())
            records_all = list(self._entries.values())

        def matches(record: GameLogRecord, tag_list: Optional[List[str]]) -> bool:
            if event_type and record.event_type != event_type:
                return False
            if not _session_matches(record, session_id):
                return False
            if since is not None and record.unix_time < since:
                return False
            if until is not None and record.unix_time > until:
                return False
            if tag_list:
                have = set(record.tags or ())
                if tag_match == "any":
                    if not any(t in have for t in tag_list):
                        return False
                else:
                    if not all(t in have for t in tag_list):
                        return False
            return True

        records = [r for r in records_all if matches(r, tags)]
        tag_matched = True
        if tags and not records:
            records = [r for r in records_all if matches(r, None)]
            tag_matched = False
        return records, tag_matched

    def recall_hybrid(
        self,
        query_text: str = "",
        query_vector: Optional[List[float]] = None,
        session_id: Optional[str] = None,
        event_type: Optional[str] = None,
        since: Optional[float] = None,
        until: Optional[float] = None,
        limit: int = 20,
        min_score: float = 0.0,
        dense_weight: float = 1.0,
        lexical_weight: float = 0.6,
        recency_decay_sec: float = 0.0,
        recency_weight: float = 1.0,
        local_boost: float = 0.0,
        role_weights: Optional[Dict[str, float]] = None,
        tags: Optional[List[str]] = None,
        tag_match: str = "all",
        entity_index: Optional[EntityIndex] = None,
        entity_boost_weight: float = 0.0,
    ) -> List[HybridRecallResult]:
        """混合召回：dense 向量 + 词法(BM25-lite) + 时效 + 角色/本地加权 + 实体加成。

        相比 `query()` 只做“单一向量余弦”，这里为“准确召回上下文”补四件事：
        1. 无向量/向量偏弱的记录仍能用词法召回（工具名/文件名/中文关键词往往很精确）；
        2. 融合时对每个子分做 min-max 归一化再加权，避免维度不可比；
        3. 支持按 role/event_type 前缀加权（如 tool/result 近因证据）、按同 session 本地加权、
           按时效半衰衰减（最近的关键决策优于久远同主题）；
        4. entity_index 非空且 entity_boost_weight>0 时启用实体层：查询实体精确命中
           已链接记忆的实体 → 实体子分（借鉴 mem0 的 entity boost）。

        返回值带各子分与命中词，便于审计与调参。
        """
        now = time.time()
        with self._lock:
            self._sweep_expired_locked(now)
            records = list(self._entries.values())

        def matches(record: GameLogRecord, tags: Optional[List[str]] = None) -> bool:
            if event_type and record.event_type != event_type:
                return False
            if not _session_matches(record, session_id):
                return False
            if since is not None and record.unix_time < since:
                return False
            if until is not None and record.unix_time > until:
                return False
            if tags:
                have = set(record.tags or ())
                if tag_match == "any":
                    if not any(t in have for t in tags):
                        return False
                else:
                    if not all(t in have for t in tags):
                        return False
            return True

        records_all = list(records)
        records = [r for r in records_all if matches(r, tags)]
        tag_matched = True
        if tags and not records:
            # 标签零命中：降级为无标签查询（与 SqliteLogStore 语义一致）
            records = [r for r in records_all if matches(r, None)]
            tag_matched = False

        # 无任何语义/文本输入时退化为“最近优先”的时序召回。
        q_tokens = _tokenize_text(query_text or "")
        dense_usable = bool(query_vector)
        if not q_tokens and not dense_usable:
            records.sort(key=lambda r: r.unix_time, reverse=True)
            return [
                HybridRecallResult(
                    record=r.model_copy(deep=True), score=0.0, tag_matched=tag_matched
                )
                for r in records[:limit]
            ]

        dense_by_id: Dict[str, float] = {}
        if dense_usable:
            with_vec = [r for r in records if r.semantic_vector]
            if with_vec:
                dim = len(query_vector)
                usable = [r for r in with_vec if len(r.semantic_vector) == dim]
                if usable:
                    X = np.asarray([r.semantic_vector for r in usable], dtype="<f4")
                    qv = np.asarray(query_vector, dtype="<f4")
                    xnorms = np.linalg.norm(X, axis=1)
                    qnorm = float(np.linalg.norm(qv))
                    if qnorm > 0.0:
                        dots = X @ qv
                        for r, dot, xn in zip(usable, dots, xnorms):
                            if xn > 0.0:
                                dense_by_id[r.event_id] = float(dot / (xn * qnorm))

        entity_hits_by_id: Dict[str, List[str]] = {}
        if entity_index is not None and entity_boost_weight > 0:
            entity_hits_by_id = entity_raw_hits(
                entity_index, query_text, (r.event_id for r in records)
            )

        scored: List[Dict[str, object]] = []
        for record in records:
            dense = dense_by_id.get(record.event_id, 0.0)
            lex_hits: List[str] = []
            lex = 0.0
            if q_tokens:
                lex, lex_hits = _lexical_score(q_tokens, record_tokens(record))
            recency = 0.0
            if recency_decay_sec and recency_decay_sec > 0:
                age = max(0.0, now - record.unix_time)
                recency = 1.0 / (1.0 + (age / recency_decay_sec) ** 2)
            entity_hits = entity_hits_by_id.get(record.event_id, [])
            scored.append(
                {
                    "record": record,
                    "dense": dense,
                    "lex": lex,
                    "recency": recency,
                    "hits": lex_hits,
                    "entity_hits": entity_hits,
                    "entity_count": len(entity_hits),
                }
            )

        if not scored:
            return []

        def _norm(key: str) -> Dict[str, float]:
            vals = [float(s[key]) for s in scored]
            lo, hi = min(vals), max(vals)
            if hi <= lo:
                return {id(s): 0.0 for s in scored}
            return {id(s): (float(s[key]) - lo) / (hi - lo) for s in scored}

        nd = _norm("dense") if dense_usable else {id(s): 0.0 for s in scored}
        nl = _norm("lex") if q_tokens else {id(s): 0.0 for s in scored}
        nr = _norm("recency") if recency_decay_sec and recency_decay_sec > 0 else {id(s): 0.0 for s in scored}
        ne = (
            _norm("entity_count")
            if entity_hits_by_id
            else {id(s): 0.0 for s in scored}
        )

        results: List[HybridRecallResult] = []
        for s in scored:
            record = s["record"]
            dense = float(s["dense"])
            lex = float(s["lex"])
            recency = float(s["recency"])
            entity_hits = list(s["entity_hits"])
            fusion = (
                dense_weight * nd[id(s)]
                + lexical_weight * nl[id(s)]
                + recency_weight * nr[id(s)]
                + entity_boost_weight * ne[id(s)]
                + (local_boost if _same_session(record, session_id) else 0.0)
                + _role_bonus(record, role_weights)
            )
            results.append(
                HybridRecallResult(
                    record=record.model_copy(deep=True),
                    score=round(fusion, 6),
                    dense_score=round(dense, 6),
                    lexical_score=round(lex, 6),
                    recency_score=round(recency, 6),
                    entity_score=round(float(ne[id(s)]), 6),
                    local=_same_session(record, session_id),
                    matched_terms=s["hits"],
                    entity_hits=entity_hits,
                    tag_matched=tag_matched,
                )
            )

        results.sort(key=lambda item: (item.score, item.record.unix_time), reverse=True)
        if min_score > 0:
            results = [r for r in results if r.score >= min_score]
        return results[:limit]

    def all(self) -> List[GameLogRecord]:
        with self._lock:
            return [record.model_copy(deep=True) for record in self._entries.values()]

    def ingest_jsonl(
        self,
        source_path: os.PathLike[str] | str,
        embedder: Optional[Callable[[GameLogRecord], List[float]]] = None,
        force_reembed: bool = False,
        limit: Optional[int] = None,
    ) -> int:
        """读取 Godot 写出的 JSONL 日志文件。

        参数:
            source_path: game_events.jsonl 路径
            embedder: 给日志生成 semantic_vector 的函数；为 None 时保留原向量/空向量
            force_reembed: True 时即使日志已有向量也重新生成
            limit: 最多读取前 N 行
        """
        source = Path(source_path).expanduser()
        if not source.exists():
            raise FileNotFoundError(f"log source not found: {source}")

        count = 0
        with source.open("r", encoding="utf-8") as f:
            for line in f:
                if limit is not None and count >= limit:
                    break
                line = line.strip()
                if not line:
                    continue
                try:
                    raw = json.loads(line)
                    record = GameLogRecord.model_validate(raw)
                except Exception:
                    # 坏行不阻塞整体导入，便于边写边读时容忍半行/乱码。
                    continue
                if record.level == "DEBUG":
                    # DEBUG 级别只留原始遥测文件，不入向量库（日志级别决定是否写库）
                    continue
                if embedder and (force_reembed or not record.semantic_vector):
                    try:
                        vector = embedder(record)
                        record.semantic_vector = [float(x) for x in vector]
                    except Exception:
                        pass
                self.add(record)
                count += 1
        return count

    def save(self, path: Optional[os.PathLike[str] | str] = None) -> None:
        """把当前内存快照完整写入 JSONL。"""
        target = Path(path).expanduser() if path else self._path
        if target is None:
            raise ValueError("save() requires a path when the store was created without one")
        with self._lock:
            target.parent.mkdir(parents=True, exist_ok=True)
            self._save_locked(target)
            self._path = target
            self._persisted_ids = set(self._entries.keys())

    # ---------- internal ----------

    def _sweep_expired_locked(self, now: float) -> int:
        """删除已过期记录。调用方必须持有 self._lock。"""
        if self._lifecycle is not None:
            # 死期+持续时间语义：deadline_base + duration_sec <= now 即删
            expired = [
                event_id
                for event_id in self._lifecycle.sweep(now=now)
                if event_id in self._entries
            ]
        else:
            retention = self._config.retention_sec
            if retention is None or retention <= 0:
                return 0
            threshold = now - retention
            expired = [
                event_id
                for event_id, record in self._entries.items()
                if record.last_used_at < threshold
            ]
        removed = 0
        for event_id in expired:
            self._entries.pop(event_id, None)
            self._persisted_ids.discard(event_id)
            removed += 1
        if removed and self._path:
            self._save_locked(self._path)
        return removed

    def _evict_over_cap_locked(self, now: float) -> int:
        """超过条数上限时，按“最久未被使用”淘汰。调用方必须持有 self._lock。"""
        max_entries = self._config.max_entries
        if max_entries is None or max_entries <= 0:
            return 0
        removed = 0
        while len(self._entries) > max_entries:
            if self._lifecycle is not None:
                # 死期语义：优先淘汰“理论删除时刻(expires_at)”最早的
                def _due(event_id: str) -> float:
                    meta = self._lifecycle.get(event_id)
                    return (meta.expires_at() if meta else 0.0, event_id)
                victim_id = min(self._entries, key=_due)
            else:
                victim_id = min(
                    self._entries,
                    key=lambda event_id: (
                        self._entries[event_id].last_used_at,
                        event_id,
                    ),
                )
            self._entries.pop(victim_id, None)
            if self._lifecycle is not None:
                self._lifecycle.delete(victim_id)
            self._persisted_ids.discard(victim_id)
            removed += 1
        if removed and self._path:
            self._save_locked(self._path)
        return removed

    def _load(self, path: Path) -> None:
        with self._lock:
            with path.open("r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        record = GameLogRecord.model_validate(json.loads(line))
                        if record.last_used_at <= 0:
                            # 旧文件没有 last_used_at：视为刚入库存活，避免一载入即被清
                            record.last_used_at = time.time()
                        self._entries[record.event_id] = record
                        self._persisted_ids.add(record.event_id)
                        if self._lifecycle is not None:
                            # 重启/加载时重建生命周期元数据：从 last_used_at 起算，
                            # 避免“载入即过期”或“无 meta 无法被生命周期清理”。
                            base = record.last_used_at or record.unix_time or time.time()
                            if self._lifecycle.get(record.event_id) is None:
                                # 用 manager 自身的默认 duration（而非 retention_sec）
                                self._lifecycle.create(record.event_id, now=base)
                    except Exception:
                        continue

    def _save_locked(self, path: Path) -> None:
        # 调用方必须持有 self._lock
        tmp_path = path.with_suffix(path.suffix + ".tmp")
        with tmp_path.open("w", encoding="utf-8") as f:
            for record in self._entries.values():
                f.write(record.model_dump_json() + "\n")
        tmp_path.replace(path)


def _cosine_similarity(left: List[float], right: List[float]) -> Optional[float]:
    if len(left) != len(right) or not left:
        return None

    dot = 0.0
    norm_left = 0.0
    norm_right = 0.0
    for a, b in zip(left, right):
        dot += a * b
        norm_left += a * a
        norm_right += b * b

    if norm_left == 0.0 or norm_right == 0.0:
        return None
    return dot / (math.sqrt(norm_left) * math.sqrt(norm_right))


def _session_matches(record: GameLogRecord, session_id: Optional[str]) -> bool:
    """None=不过滤；非空时全局记录(空 session_id)与同 session 记录都可命中。"""
    if not session_id:
        return True
    return record.session_id in ("", session_id)


def _same_session(record: GameLogRecord, session_id: Optional[str]) -> bool:
    return bool(session_id) and record.session_id == session_id


def _tokenize_text(text: str) -> List[str]:
    """轻量分词：ASCII 词 + 中文单字/双字组合（与 HashEmbedder 同思路）。

    用于词法召回（工具名/文件名/中文关键词往往是很精确的信号），不追求 NLP 完整度。
    """
    out: List[str] = list(re.findall(r"[a-zA-Z0-9_]+", text.lower()))
    for phrase in re.findall(r"[\u4e00-\u9fff]+", text):
        if len(phrase) == 1:
            out.append(phrase)
            continue
        out.append(phrase)
        for i in range(len(phrase)):
            out.append(phrase[i])
            if i + 1 < len(phrase):
                out.append(phrase[i : i + 2])
    return out


# 分词写入缓存: key=event_id, 校验 embedding 文本指纹, 命中则零成本。
# 查询热点路径的分词占大头(实测 3.3k 候选 ~0.4s/次), 缓存后亚毫秒。
_TOKEN_CACHE: Dict[str, tuple] = {}
_TOKEN_CACHE_CAP = 50000


def record_tokens(record: GameLogRecord) -> List[str]:
    """带缓存的分词: 结果与 _tokenize_text(record.to_embedding_text()) 完全一致。"""
    text = record.to_embedding_text()
    cached = _TOKEN_CACHE.get(record.event_id)
    if cached is not None and cached[0] == text:
        return cached[1]
    tokens = _tokenize_text(text)
    if len(_TOKEN_CACHE) >= _TOKEN_CACHE_CAP:
        _TOKEN_CACHE.clear()  # 简单防膨胀; 正常远达不到
    _TOKEN_CACHE[record.event_id] = (text, tokens)
    return tokens


def _lexical_score(
    query_tokens: List[str],
    record_tokens: List[str],
) -> tuple[float, List[str]]:
    """BM25-lite 的词法重叠分：命中词越重要(词频)、越多的记录分越高。"""
    if not query_tokens:
        return 0.0, []
    q = Counter(query_tokens)
    r = Counter(record_tokens)
    hits: List[str] = []
    acc = 0.0
    for token, qf in q.items():
        rf = r.get(token, 0)
        if rf <= 0:
            continue
        hits.append(token)
        acc += math.log1p(rf) * (1.0 + math.log1p(qf))
    score = acc / (1.0 + len(query_tokens)) if query_tokens else 0.0
    return round(score, 6), sorted(set(hits))


def _role_bonus(
    record: GameLogRecord,
    role_weights: Optional[Dict[str, float]],
) -> float:
    """按 kind/event_type 加权：支持精确 key 与 '前缀*' 通配（如 'tool/*'）。"""
    if not role_weights:
        return 0.0
    key = record.kind or record.event_type
    if not key:
        return 0.0
    exact = role_weights.get(key)
    if exact is not None:
        return exact
    for pattern, weight in role_weights.items():
        if pattern.endswith("*") and key.startswith(pattern[:-1]):
            return weight
    return 0.0


def fuse_hybrid(
    records: List[GameLogRecord],
    *,
    now: float,
    query_text: str = "",
    query_vector: Optional[List[float]] = None,
    session_id: Optional[str] = None,
    limit: int = 20,
    min_score: float = 0.0,
    dense_weight: float = 1.0,
    lexical_weight: float = 0.6,
    recency_decay_sec: float = 0.0,
    recency_weight: float = 1.0,
    local_boost: float = 0.0,
    role_weights: Optional[Dict[str, float]] = None,
    tag_matched: bool = True,
    prefilter_top: int = 0,
    entity_index: Optional[EntityIndex] = None,
    entity_boost_weight: float = 0.0,
) -> List["HybridRecallResult"]:
    """混合融合打分(模块级, 单库/多库共用)。

    prefilter_top > 0 时启用宽查询保护: 候选超过该数且带查询向量,
    先按 dense 分数保留 top-N 再做词法精排(词法是逐候选成本, 预筛兜住 15s 级爆点)。
    注意: 预筛会丢弃无向量候选, 仅建议在语义检索场景开启。

    语义与 LogVectorStore.recall_hybrid 完全一致: dense 余弦(numpy 批量)
    + BM25-lite 词法 + recency, min-max 归一化加权 + local/role 加权;
    entity_index/entity_boost_weight 非空时再加实体加成(同 recall_hybrid)。
    parity 由测试守护(test_multi_log_store.py)。
    """
    q_tokens = _tokenize_text(query_text or "")
    dense_usable = bool(query_vector)
    if not q_tokens and not dense_usable:
        records = sorted(records, key=lambda r: r.unix_time, reverse=True)
        return [
            HybridRecallResult(record=r, score=0.0, tag_matched=tag_matched)
            for r in records[:limit]
        ]

    dense_by_id: Dict[str, float] = {}
    if dense_usable:
        with_vec = [r for r in records if r.semantic_vector]
        if with_vec:
            dim = len(query_vector)
            usable = [r for r in with_vec if len(r.semantic_vector) == dim]
            if usable:
                X = np.asarray([r.semantic_vector for r in usable], dtype="<f4")
                qv = np.asarray(query_vector, dtype="<f4")
                xnorms = np.linalg.norm(X, axis=1)
                qnorm = float(np.linalg.norm(qv))
                if qnorm > 0.0:
                    dots = X @ qv
                    for r, dot, xn in zip(usable, dots, xnorms):
                        if xn > 0.0:
                            dense_by_id[r.event_id] = float(dot / (xn * qnorm))

    if prefilter_top and prefilter_top > 0 and dense_usable and len(records) > prefilter_top:
        records = sorted(
            records,
            key=lambda r: dense_by_id.get(r.event_id, float("-inf")),
            reverse=True,
        )[:prefilter_top]

    entity_hits_by_id: Dict[str, List[str]] = {}
    if entity_index is not None and entity_boost_weight > 0:
        entity_hits_by_id = entity_raw_hits(
            entity_index, query_text, (r.event_id for r in records)
        )

    scored: List[Dict[str, object]] = []
    for record in records:
        dense = dense_by_id.get(record.event_id, 0.0)
        lex, hits = 0.0, []
        if q_tokens:
            lex, hits = _lexical_score(q_tokens, record_tokens(record))
        recency = 0.0
        if recency_decay_sec and recency_decay_sec > 0:
            age = max(0.0, now - record.unix_time)
            recency = 1.0 / (1.0 + (age / recency_decay_sec) ** 2)
        entity_hits = entity_hits_by_id.get(record.event_id, [])
        scored.append(
            {
                "record": record,
                "dense": dense,
                "lex": lex,
                "recency": recency,
                "hits": hits,
                "entity_hits": entity_hits,
                "entity_count": len(entity_hits),
            }
        )
    if not scored:
        return []

    def _norm(key: str) -> Dict[int, float]:
        vals = [float(s[key]) for s in scored]
        lo, hi = min(vals), max(vals)
        if hi <= lo:
            return {i: 0.0 for i in range(len(scored))}
        return {i: (vals[i] - lo) / (hi - lo) for i in range(len(scored))}

    nd, nl, nr = _norm("dense"), _norm("lex"), _norm("recency")
    ne = _norm("entity_count") if entity_hits_by_id else {i: 0.0 for i in range(len(scored))}
    results: List[HybridRecallResult] = []
    for i, s_ in enumerate(scored):
        record: GameLogRecord = s_["record"]  # type: ignore[assignment]
        fusion = (
            dense_weight * nd[i]
            + lexical_weight * nl[i]
            + recency_weight * nr[i]
            + entity_boost_weight * ne[i]
            + (local_boost if _same_session(record, session_id) else 0.0)
            + _role_bonus(record, role_weights)
        )
        results.append(
            HybridRecallResult(
                record=record,
                score=round(fusion, 6),
                dense_score=round(float(s_["dense"]), 6),
                lexical_score=round(float(s_["lex"]), 6),
                recency_score=round(float(s_["recency"]), 6),
                entity_score=round(float(ne[i]), 6),
                local=_same_session(record, session_id),
                matched_terms=list(s_["hits"]),  # type: ignore[arg-type]
                entity_hits=list(s_["entity_hits"]),  # type: ignore[arg-type]
                tag_matched=tag_matched,
            )
        )
    results.sort(key=lambda item: (item.score, item.record.unix_time), reverse=True)
    if min_score > 0:
        results = [r for r in results if r.score >= min_score]
    return results[:limit]
