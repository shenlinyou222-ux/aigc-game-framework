# -*- coding: utf-8 -*-
"""把现有 JSONL 日志向量库迁移到 SQLite 单文件后端，并做双后端召回一致性校验。

一致性校验用冻结时钟：两个后端在同一个人为时间点打分，消除真实时间差带来的
recency 分漂移，保证逐 id 严格可比。

用法（默认迁移主上下文库）:
    python migrate_store_to_sqlite.py
可选:
    --source dsh_context_store.jsonl   源 JSONL
    --db      dsh_context_store.db     目标 SQLite（已存在则覆盖同 id 记录）
    --limit N                          只迁移前 N 条
    --queries N                        一致性抽检查询数（默认 12）
    --no-parity                        跳过一致性校验
"""

from __future__ import annotations

import argparse
import json
import random
import time as _real_time
from pathlib import Path

import aigc_core.log_vector_store as lvs
import aigc_core.sqlite_log_store as sls
from aigc_core.lifecycle import VectorLifecycleManager
from aigc_core.log_vector_store import LogVectorStore
from aigc_core.models import GameLogRecord, LogVectorQuery
from aigc_core.sqlite_log_store import SqliteLogStore

DEFAULT_DURATION_SEC = 7 * 24 * 3600


class _FrozenTime:
    """冻结时钟：parity 期间两个后端拿到同一个 now。"""

    def __init__(self, t: float) -> None:
        self.t = t

    def time(self) -> float:
        return self.t


def derive_tags(record: GameLogRecord) -> list:
    """迁移期自动打标（受控词表）：type:/kind:/mem: 前缀。"""
    tags = ["type:" + record.event_type]
    if record.kind:
        tags.append("kind:" + record.kind)
    tags.append("mem:global" if not record.session_id else "mem:session")
    return tags


def main() -> None:
    parser = argparse.ArgumentParser(description="Migrate JSONL log store to SQLite backend")
    parser.add_argument("--source", default="dsh_context_store.jsonl")
    parser.add_argument("--db", default="dsh_context_store.db")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--queries", type=int, default=12)
    parser.add_argument("--no-parity", action="store_true")
    args = parser.parse_args()

    source = Path(args.source)
    db = Path(args.db)
    assert source.exists(), "source missing: %s" % source

    lm_old = VectorLifecycleManager(default_duration_sec=DEFAULT_DURATION_SEC)
    old = LogVectorStore(path=source, lifecycle=lm_old)
    print("source records:", old.count())

    lm_new = VectorLifecycleManager(default_duration_sec=DEFAULT_DURATION_SEC)
    new = SqliteLogStore(path=db, lifecycle=lm_new)
    records = old.all()
    if args.limit:
        records = records[: args.limit]
    for r in records:
        r.tags = derive_tags(r)
    # 一致性前提：两库持有完全相同的记录（含 tags）。老库 add 已存在 id 只更新内存，
    # 不会回写 JSONL 源文件。
    for r in records:
        old.add(r)
    t0 = _real_time.perf_counter()
    migrated = new.add_many(records)
    dt = _real_time.perf_counter() - t0
    print("migrated: %s in %.2fs -> %s (%d bytes)" % (migrated, dt, db, db.stat().st_size))

    if not args.no_parity:
        rng = random.Random(42)
        samples = rng.sample(records, min(args.queries, len(records)))
        frozen = _FrozenTime(1788600000.0)
        lvs.time, sls.time = frozen, frozen
        mismatches = 0
        t_old = t_new = 0.0
        for seed in samples:
            data_text = json.dumps(seed.data, ensure_ascii=False)[:40]
            kw = dict(
                query_text="%s %s %s" % (seed.event_type, seed.kind, data_text),
                session_id=None,
                limit=8,
                recency_decay_sec=24 * 3600.0,
                role_weights={"tool/*": 0.2},
            )
            t1 = _real_time.perf_counter()
            ra = old.recall_hybrid(**kw)
            t_old += _real_time.perf_counter() - t1
            t1 = _real_time.perf_counter()
            rb = new.recall_hybrid(**kw)
            t_new += _real_time.perf_counter() - t1
            if [x.record.event_id for x in ra] != [x.record.event_id for x in rb]:
                mismatches += 1
                print("  MISMATCH seed=", seed.event_id)
                print("    jsonl :", [x.record.event_id for x in ra][:4])
                print("    sqlite:", [x.record.event_id for x in rb][:4])
        lvs.time, sls.time = _real_time, _real_time
        print("parity (frozen clock): %d/%d identical" % (args.queries - mismatches, args.queries))
        print("latency (%d queries, all-candidates scan): jsonl=%.3fs  sqlite=%.3fs" % (args.queries, t_old, t_new))

    for tag in ("type:tool/result", "type:assistant/message", "mem:global"):
        n = len(new.query(LogVectorQuery(tags=[tag], limit=500)))
        print("tag gate %s: %d records" % (tag, n))
    new.close()
    print("MIGRATION DONE")


if __name__ == "__main__":
    main()
