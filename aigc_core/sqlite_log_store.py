# -*- coding: utf-8 -*-
"""SqliteLogStore: LogVectorStore 的单文件 SQLite 后端（标签精确门控 + numpy 语义扫描）。

与 LogVectorStore 的关系：对外方法面（add/query/recall_hybrid/生命周期/容量）语义对齐，
打分函数直接复用 log_vector_store 的模块级实现（BM25-lite 词法、min-max 归一化融合、
role/session 加权），保证两后端召回结果一致；差异只在存储与扫描层：

- 存储：单 .db 文件（records + record_tags 倒排表），不再常驻全量 Python 对象；
- 标签精确匹配：SQL 门控（all=AND / any=OR），先于语义扫描；标签零命中自动降级为
  无标签查询，结果带 tag_matched=False 供调用方审计；
- 语义模糊匹配：门控后的候选集上 numpy 矩阵余弦（float32），复杂度 O(候选数) 而非 O(全库)；
- 生命周期：直接复用 VectorLifecycleManager（内存元数据 + SQL 删除执行），语义不变。

FTS5 暂不引入：词法分在本设计里是“融合信号”而非“召回门”（需对全部候选打分后做
min-max 归一化），用 FTS5 的 BM25 替换自研 BM25-lite 会改变融合语义。
"""

from __future__ import annotations

import json
import math
import os
import sqlite3
import threading
import time
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple

import numpy as np

from .log_vector_store import (
    StoreLifecycleConfig,
    _cosine_similarity,
    _lexical_score,
    _role_bonus,
    _same_session,
    _session_matches,
    _tokenize_text,
    record_tokens,
    load_lifecycle_config,
    write_lifecycle_config,
)
from .lifecycle import VectorLifecycleManager
from .entity_index import EntityIndex, entity_raw_hits
from .models import GameLogRecord, LogSearchResult, LogVectorQuery, HybridRecallResult

_SCHEMA = """
CREATE TABLE IF NOT EXISTS records (
    event_id        TEXT PRIMARY KEY,
    event_type      TEXT NOT NULL DEFAULT '',
    kind            TEXT NOT NULL DEFAULT '',
    session_id      TEXT NOT NULL DEFAULT '',
    time            TEXT NOT NULL DEFAULT '',
    unix_time       REAL NOT NULL DEFAULT 0,
    frame           INTEGER NOT NULL DEFAULT 0,
    data_json       TEXT NOT NULL DEFAULT '{}',
    semantic_vector BLOB,
    last_used_at    REAL NOT NULL DEFAULT 0,
    level           TEXT NOT NULL DEFAULT 'INFO',
    schema_v        TEXT NOT NULL DEFAULT ''
);
CREATE INDEX IF NOT EXISTS idx_records_type      ON records(event_type);
CREATE INDEX IF NOT EXISTS idx_records_session   ON records(session_id);
CREATE INDEX IF NOT EXISTS idx_records_unix      ON records(unix_time);
CREATE INDEX IF NOT EXISTS idx_records_lastused  ON records(last_used_at);
CREATE TABLE IF NOT EXISTS record_tags (
    event_id TEXT NOT NULL REFERENCES records(event_id) ON DELETE CASCADE,
    tag      TEXT NOT NULL,
    PRIMARY KEY (event_id, tag)
) WITHOUT ROWID;
CREATE INDEX IF NOT EXISTS idx_tags_tag ON record_tags(tag);
"""


class SqliteLogStore:
    """LogVectorStore 的 SQLite 单文件后端，方法面语义对齐（见模块 docstring）。"""

    def __init__(
        self,
        path: Optional[os.PathLike[str] | str] = None,
        config: Optional[StoreLifecycleConfig] = None,
        config_path: Optional[os.PathLike[str] | str] = None,
        lifecycle: Optional[VectorLifecycleManager] = None,
    ) -> None:
        if path is None:
            raise ValueError("SqliteLogStore requires a .db file path")
        self._path = Path(path).expanduser()
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._lifecycle = lifecycle
        if config is not None:
            self._config = config
        elif config_path is not None:
            p = Path(config_path).expanduser()
            if not p.exists():
                write_lifecycle_config(p)
            self._config = load_lifecycle_config(p)
        else:
            self._config = load_lifecycle_config()
        self._conn = sqlite3.connect(str(self._path), check_same_thread=False)
        self._conn.execute("PRAGMA foreign_keys=ON")
        self._conn.executescript(_SCHEMA)
        # 老库自动迁移: 补 level/schema_v 列
        cols = [r[1] for r in self._conn.execute("PRAGMA table_info(records)")]
        if "level" not in cols:
            self._conn.execute("ALTER TABLE records ADD COLUMN level TEXT NOT NULL DEFAULT 'INFO'")
        if "schema_v" not in cols:
            self._conn.execute("ALTER TABLE records ADD COLUMN schema_v TEXT NOT NULL DEFAULT ''")
        self._conn.commit()
        self._load_lifecycle_meta()

    # ---------- 构造/加载 ----------

    def _load_lifecycle_meta(self) -> None:
        """与 LogVectorStore._load 对齐：重启后从 last_used_at 重建生命周期元数据。"""
        if self._lifecycle is None:
            return
        now = time.time()
        for event_id, last_used, unix in self._conn.execute(
            "SELECT event_id, last_used_at, unix_time FROM records"
        ):
            if self._lifecycle.get(event_id) is None:
                base = last_used or unix or now
                self._lifecycle.create(event_id, now=base)

    @property
    def path(self) -> Path:
        return self._path

    @property
    def config(self) -> StoreLifecycleConfig:
        return self._config

    @property
    def lifecycle(self) -> Optional[VectorLifecycleManager]:
        return self._lifecycle

    def close(self) -> None:
        with self._lock:
            self._conn.close()

    # ---------- 行 <-> 模型 ----------

    @staticmethod
    def _row_to_record(row: Tuple) -> GameLogRecord:
        (
            event_id, event_type, kind, session_id, time_str, unix_time,
            frame, data_json, vec_blob, last_used_at, level, schema_v,
        ) = row
        record = GameLogRecord(
            event_id=event_id,
            event_type=event_type,
            time=time_str,
            unix_time=unix_time,
            frame=frame,
            data=json.loads(data_json),
            semantic_vector=(
                list(np.frombuffer(vec_blob, dtype="<f4")) if vec_blob else []
            ),
            last_used_at=last_used_at,
            session_id=session_id,
            kind=kind,
            level=level,
            schema_v=schema_v,
        )
        return record

    def _fetch_records(
        self, where: str, params: Tuple, tags: Optional[List[str]], tag_match: str
    ) -> Tuple[List[GameLogRecord], bool]:
        """按元数据门控取候选记录。返回 (records, tag_gate_hit)。

        tag_gate_hit=False 表示带了标签但门控零命中（已自动降级重查）。
        """
        rows = self._fetch_rows(where, params, tags, tag_match)
        gate_requested = bool(tags)
        tag_matched = True
        if gate_requested and not rows:
            # 降级：去掉标签门控重查，保证“精确不命中也不空手”。
            rows = self._fetch_rows(where, params, None, tag_match)
            tag_matched = False
        tagmap = self._batch_tags([r[0] for r in rows])
        return [self._record_from_row(r, tagmap) for r in rows], tag_matched

    def _batch_tags(self, event_ids: List[str]) -> Dict[str, List[str]]:
        """一次查询批量取候选集标签，避免逐条 N+1。"""
        if not event_ids:
            return {}
        ph = ",".join("?" for _ in event_ids)
        tagmap: Dict[str, List[str]] = {}
        for eid, tag in self._conn.execute(
            f"SELECT event_id, tag FROM record_tags WHERE event_id IN ({ph})",
            list(event_ids),
        ):
            tagmap.setdefault(eid, []).append(tag)
        for v in tagmap.values():
            v.sort()
        return tagmap

    def _fetch_rows(
        self, where: str, params: Tuple, tags: Optional[List[str]], tag_match: str
    ) -> List[Tuple]:
        tag_sql = ""
        if tags:
            ph = ",".join("?" for _ in tags)
            if tag_match == "any":
                tag_sql = (
                    f" AND event_id IN (SELECT event_id FROM record_tags WHERE tag IN ({ph}))"
                )
            else:  # all
                tag_sql = (
                    f" AND event_id IN ("
                    f"SELECT event_id FROM record_tags WHERE tag IN ({ph}) "
                    f"GROUP BY event_id HAVING COUNT(DISTINCT tag) = {len(tags)})"
                )
        sql = (
            "SELECT event_id, event_type, kind, session_id, time, unix_time, frame, "
            "data_json, semantic_vector, last_used_at, level, schema_v FROM records WHERE " + where + tag_sql
        )
        return self._conn.execute(sql, list(params) + list(tags or [])).fetchall()

    def _record_from_row(
        self, row: Tuple, tagmap: Optional[Dict[str, List[str]]] = None
    ) -> GameLogRecord:
        record = self._row_to_record(row)
        if tagmap is not None:
            record.tags = tagmap.get(record.event_id, [])
        else:
            record.tags = self._tags_of(record.event_id)
        return record

    def _tags_of(self, event_id: str) -> List[str]:
        return [
            r[0]
            for r in self._conn.execute(
                "SELECT tag FROM record_tags WHERE event_id=? ORDER BY tag", (event_id,)
            )
        ]

    # ---------- 写入 ----------

    def add(self, record: GameLogRecord) -> GameLogRecord:
        now = time.time()
        with self._lock:
            stored = record.model_copy(deep=True)
            if stored.last_used_at <= 0:
                stored.last_used_at = now
            is_new = self._conn.execute(
                "SELECT 1 FROM records WHERE event_id=?", (stored.event_id,)
            ).fetchone() is None
            self._upsert(stored)
            if self._lifecycle is not None:
                if is_new:
                    self._lifecycle.create(stored.event_id, now=now)
                else:
                    self._lifecycle.touch(stored.event_id, now=now)
            self._conn.commit()
            self._sweep_expired_locked(now)
            self._evict_over_cap_locked(now)
            return record

    def add_many(self, records: Iterable[GameLogRecord]) -> int:
        """批量写入：单事务提交，末尾统一懒维护（最终状态与逐条 add 一致）。"""
        now = time.time()
        count = 0
        with self._lock:
            for record in records:
                stored = record.model_copy(deep=True)
                if stored.last_used_at <= 0:
                    stored.last_used_at = now
                is_new = self._conn.execute(
                    "SELECT 1 FROM records WHERE event_id=?", (stored.event_id,)
                ).fetchone() is None
                self._upsert(stored)
                if self._lifecycle is not None:
                    if is_new:
                        self._lifecycle.create(stored.event_id, now=now)
                    else:
                        self._lifecycle.touch(stored.event_id, now=now)
                count += 1
            self._conn.commit()
            self._sweep_expired_locked(now)
            self._evict_over_cap_locked(now)
        return count

    def _upsert(self, stored: GameLogRecord) -> None:
        vec_blob = (
            np.asarray(stored.semantic_vector, dtype="<f4").tobytes()
            if stored.semantic_vector
            else None
        )
        self._conn.execute(
            "INSERT INTO records (event_id, event_type, kind, session_id, time, unix_time,"
            " frame, data_json, semantic_vector, last_used_at, level, schema_v)"
            " VALUES (?,?,?,?,?,?,?,?,?,?,?,?)"
            " ON CONFLICT(event_id) DO UPDATE SET event_type=excluded.event_type,"
            " kind=excluded.kind, session_id=excluded.session_id, time=excluded.time,"
            " unix_time=excluded.unix_time, frame=excluded.frame,"
            " data_json=excluded.data_json, semantic_vector=excluded.semantic_vector,"
            " last_used_at=excluded.last_used_at, level=excluded.level,"
            " schema_v=excluded.schema_v",
            (
                stored.event_id, stored.event_type, stored.kind, stored.session_id,
                stored.time, stored.unix_time, stored.frame,
                json.dumps(stored.data, ensure_ascii=False), vec_blob, stored.last_used_at,
                stored.level, stored.schema_v,
            ),
        )
        self._conn.execute("DELETE FROM record_tags WHERE event_id=?", (stored.event_id,))
        if stored.tags:
            self._conn.executemany(
                "INSERT OR IGNORE INTO record_tags (event_id, tag) VALUES (?,?)",
                [(stored.event_id, t) for t in dict.fromkeys(stored.tags)],
            )

    def delete(self, event_id: str) -> bool:
        with self._lock:
            cur = self._conn.execute("DELETE FROM records WHERE event_id=?", (event_id,))
            self._conn.commit()
            existed = cur.rowcount > 0
            if existed and self._lifecycle is not None:
                self._lifecycle.delete(event_id)
            return existed

    def touch(self, event_id: str, at: Optional[float] = None) -> bool:
        now = time.time() if at is None else float(at)
        with self._lock:
            cur = self._conn.execute(
                "UPDATE records SET last_used_at=? WHERE event_id=?", (now, event_id)
            )
            self._conn.commit()
            if cur.rowcount > 0 and self._lifecycle is not None:
                self._lifecycle.touch(event_id, now=now)
            return cur.rowcount > 0

    # ---------- 读取 ----------

    def count(self) -> int:
        with self._lock:
            return self._conn.execute("SELECT COUNT(*) FROM records").fetchone()[0]

    def get(self, event_id: str) -> Optional[GameLogRecord]:
        with self._lock:
            row = self._conn.execute(
                "SELECT event_id, event_type, kind, session_id, time, unix_time, frame,"
                " data_json, semantic_vector, last_used_at, level, schema_v FROM records WHERE event_id=?",
                (event_id,),
            ).fetchone()
            return self._record_from_row(row, None) if row else None

    def all(self) -> List[GameLogRecord]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT event_id, event_type, kind, session_id, time, unix_time, frame,"
                " data_json, semantic_vector, last_used_at, level, schema_v FROM records ORDER BY rowid"
            ).fetchall()
            return [self._record_from_row(r, None) for r in rows]

    # ---------- 生命周期 ----------

    def lifecycle_meta(self, event_id: str) -> Optional[Dict[str, object]]:
        if self._lifecycle is None:
            return None
        meta = self._lifecycle.get(event_id)
        return meta.to_dict() if meta else None

    def lifecycle_expires_at(self, event_id: str) -> Optional[float]:
        if self._lifecycle is None:
            return None
        return self._lifecycle.expires_at(event_id)

    def set_lifecycle_duration(self, event_id: str, duration_sec: float) -> Optional[float]:
        if self._lifecycle is None:
            return None
        meta = self._lifecycle.set_duration(event_id, duration_sec)
        return meta.expires_at() if meta else None

    def sweep_expired(self, now: Optional[float] = None) -> int:
        with self._lock:
            return self._sweep_expired_locked(time.time() if now is None else float(now))

    def evict_over_cap(self, now: Optional[float] = None) -> int:
        with self._lock:
            return self._evict_over_cap_locked(time.time() if now is None else float(now))

    def warning_breached(self) -> bool:
        with self._lock:
            return self._config.max_entries > 0 and (
                self.count() > self._config.warn_entries
            )

    def capacity_status(self) -> Dict[str, object]:
        with self._lock:
            n = self.count()
            vec_n = self._conn.execute(
                "SELECT COUNT(*) FROM records WHERE semantic_vector IS NOT NULL"
            ).fetchone()[0]
            return {
                "count": n,
                "max_entries": self._config.max_entries,
                "warn_entries": self._config.warn_entries,
                "over_warning": self._config.max_entries > 0 and n > self._config.warn_entries,
                "retention_sec": self._config.retention_sec,
                "backend": "sqlite",
                "vec_count": vec_n,
                "db_path": str(self._path),
            }

    def _sweep_expired_locked(self, now: float) -> int:
        if self._lifecycle is not None:
            expired = list(self._lifecycle.sweep(now=now))
        else:
            retention = self._config.retention_sec
            if retention is None or retention <= 0:
                return 0
            rows = self._conn.execute(
                "SELECT event_id FROM records WHERE last_used_at < ?",
                (now - retention,),
            ).fetchall()
            expired = [r[0] for r in rows]
        removed = 0
        for event_id in expired:
            cur = self._conn.execute("DELETE FROM records WHERE event_id=?", (event_id,))
            removed += cur.rowcount if cur.rowcount > 0 else 0
        if removed:
            self._conn.commit()
        return removed

    def _evict_over_cap_locked(self, now: float) -> int:
        max_entries = self._config.max_entries
        if max_entries is None or max_entries <= 0:
            return 0
        removed = 0
        while self.count() > max_entries:
            if self._lifecycle is not None:
                ids = [
                    r[0]
                    for r in self._conn.execute("SELECT event_id FROM records")
                ]

                def _due(event_id: str) -> Tuple[float, str]:
                    meta = self._lifecycle.get(event_id)
                    return (meta.expires_at() if meta else 0.0, event_id)

                victim_id = min(ids, key=_due)
            else:
                row = self._conn.execute(
                    "SELECT event_id FROM records"
                    " ORDER BY last_used_at ASC, event_id ASC LIMIT 1"
                ).fetchone()
                if row is None:
                    break
                victim_id = row[0]
            self._conn.execute("DELETE FROM records WHERE event_id=?", (victim_id,))
            if self._lifecycle is not None:
                self._lifecycle.delete(victim_id)
            removed += 1
        if removed:
            self._conn.commit()
        return removed

    # ---------- 查询 ----------

    def query(self, query: LogVectorQuery) -> List[LogSearchResult]:
        """语义与 LogVectorStore.query 对齐；额外支持 tags 精确门控与零命中降级。"""
        now = time.time()
        with self._lock:
            self._sweep_expired_locked(now)
            where, params = self._meta_where(query.event_type, query.session_id, query.since, query.until)
            candidates, tag_matched = self._fetch_records(
                where, params, query.tags, query.tag_match
            )

        if query.query_vector:
            qv = np.asarray(query.query_vector, dtype="<f4")
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
            LogSearchResult(record=record, tag_matched=tag_matched)
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
    ) -> tuple:
        """按元数据(含标签门控/零命中降级)拉取候选, 供多库扇出复用。"""
        with self._lock:
            self._sweep_expired_locked(time.time())
            where, params = self._meta_where(event_type, session_id, since, until)
            return self._fetch_records(where, params, tags, tag_match)

    @staticmethod
    def _meta_where(
        event_type: Optional[str],
        session_id: Optional[str],
        since: Optional[float],
        until: Optional[float],
    ) -> Tuple[str, Tuple]:
        conds: List[str] = ["1=1"]
        params: List[Any] = []
        if event_type:
            conds.append("event_type = ?")
            params.append(event_type)
        if session_id:
            conds.append("session_id IN ('', ?)")
            params.append(session_id)
        if since is not None:
            conds.append("unix_time >= ?")
            params.append(since)
        if until is not None:
            conds.append("unix_time <= ?")
            params.append(until)
        return " AND ".join(conds), tuple(params)

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
        """混合召回：语义与 LogVectorStore.recall_hybrid 对齐 + 标签精确门控/降级 + 实体加成。

        打分函数直接复用 log_vector_store 的模块级实现，保证两后端分数一致。
        """
        now = time.time()
        with self._lock:
            self._sweep_expired_locked(now)
            where, params = self._meta_where(event_type, session_id, since, until)
            records, tag_matched = self._fetch_records(where, params, tags, tag_match)

        q_tokens = _tokenize_text(query_text or "")
        dense_usable = bool(query_vector)
        if not q_tokens and not dense_usable:
            records.sort(key=lambda r: r.unix_time, reverse=True)
            return [
                HybridRecallResult(record=r, score=0.0, tag_matched=tag_matched)
                for r in records[:limit]
            ]

        # dense 子分：候选集上有向量的记录做 numpy 矩阵余弦（无向量=0，与原实现一致）
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
                rec_tokens = record_tokens(record)
                lex, lex_hits = _lexical_score(q_tokens, rec_tokens)
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

        def _norm(key: str) -> Dict[int, float]:
            vals = [float(s[key]) for s in scored]
            lo, hi = min(vals), max(vals)
            if hi <= lo:
                return {i: 0.0 for i in range(len(scored))}
            return {i: (vals[i] - lo) / (hi - lo) for i in range(len(scored))}

        nd = _norm("dense")
        nl = _norm("lex")
        nr = _norm("recency")
        ne = _norm("entity_count") if entity_hits_by_id else {i: 0.0 for i in range(len(scored))}

        results: List[HybridRecallResult] = []
        for i, s in enumerate(scored):
            record: GameLogRecord = s["record"]  # type: ignore[assignment]
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
                    dense_score=round(float(s["dense"]), 6),
                    lexical_score=round(float(s["lex"]), 6),
                    recency_score=round(float(s["recency"]), 6),
                    entity_score=round(float(ne[i]), 6),
                    local=_same_session(record, session_id),
                    matched_terms=list(s["hits"]),  # type: ignore[arg-type]
                    entity_hits=list(s["entity_hits"]),  # type: ignore[arg-type]
                    tag_matched=tag_matched,
                )
            )

        results.sort(key=lambda item: (item.score, item.record.unix_time), reverse=True)
        if min_score > 0:
            results = [r for r in results if r.score >= min_score]
        return results[:limit]

    # ---------- 导入/导出 ----------

    def ingest_jsonl(
        self,
        source_path: os.PathLike[str] | str,
        embedder: Optional[Callable[[GameLogRecord], List[float]]] = None,
        force_reembed: bool = False,
        limit: Optional[int] = None,
    ) -> int:
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
                    record = GameLogRecord.model_validate(json.loads(line))
                except Exception:
                    continue
                if record.level == "DEBUG":
                    # DEBUG 级别只留原始遥测文件，不入向量库（与 jsonl 后端语义一致）
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
        """SQLite 后端为落库式写入；save() 仅做 commit（+可选 VACUUM 压缩）。"""
        with self._lock:
            self._conn.commit()
            if path is not None and Path(path).expanduser() != self._path:
                raise ValueError(
                    "SqliteLogStore.save() does not support exporting to another path;"
                    " create a new store and ingest instead"
                )
            self._conn.execute("VACUUM")
            self._conn.commit()
