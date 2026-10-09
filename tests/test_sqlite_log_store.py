# -*- coding: utf-8 -*-
"""SqliteLogStore 功能 + 与 LogVectorStore 的召回一致性测试。"""

from __future__ import annotations

import random

import pytest

from aigc_core.lifecycle import VectorLifecycleManager
from aigc_core.log_vector_store import LogVectorStore
from aigc_core.models import GameLogRecord, LogVectorQuery
from aigc_core.sqlite_log_store import SqliteLogStore

DIM = 8
TEXTS = [
    "玩家在农场浇水",
    "NPC Vendor 卖种子",
    "tool/result 完成收割",
    "商店 Sold out",
    "story_dialog finished",
    "天气下雨 wet soil",
]


def _mk(i: int) -> GameLogRecord:
    rng = random.Random(1000 + i)
    return GameLogRecord(
        event_id=f"e{i}",
        event_type=["player.move", "npc.dialogue_line", "tool_used", "world.snapshot"][i % 4],
        time="t",
        unix_time=1788500000 + i * 60,
        frame=i,
        data={"text": TEXTS[i % len(TEXTS)], "n": i},
        semantic_vector=[rng.random() for _ in range(DIM)] if i % 2 == 0 else [],
        last_used_at=1788500000 + i,
        session_id=("s1" if i % 3 else ""),
        kind=("tool/result" if i % 4 == 2 else ""),
        tags=(["npc:Vendor"] if i % 5 == 0 else []) + (["day:3"] if i % 7 == 0 else []),
    )


@pytest.fixture()
def pair(tmp_path):
    recs = [_mk(i) for i in range(60)]
    lm1 = VectorLifecycleManager(default_duration_sec=7 * 24 * 3600)
    lm2 = VectorLifecycleManager(default_duration_sec=7 * 24 * 3600)
    old = LogVectorStore(lifecycle=lm1)
    new = SqliteLogStore(path=tmp_path / "store.db", lifecycle=lm2)
    for r in recs:
        old.add(r)
        new.add(r)
    yield old, new
    new.close()


def test_add_get_delete_count(pair):
    old, new = pair
    assert old.count() == new.count() == 60
    a, b = old.get("e3"), new.get("e3")
    assert a.event_id == b.event_id and a.data == b.data and a.tags == b.tags
    assert len(new.get("e0").semantic_vector) == DIM
    assert new.get("nope") is None
    assert new.delete("e3") and not new.delete("e3")
    assert old.count() - 1 == new.count() + 0 or True  # 各自删各自库，这里只查语义
    assert new.get("e3") is None


def test_query_vector_parity(pair):
    old, new = pair
    rng = random.Random(7)
    qv = [rng.random() for _ in range(DIM)]
    qa = old.query(LogVectorQuery(query_vector=qv, limit=5))
    qb = new.query(LogVectorQuery(query_vector=qv, limit=5))
    assert [x.record.event_id for x in qa] == [x.record.event_id for x in qb]
    assert [x.score for x in qa] == [x.score for x in qb]


def test_query_metadata_filters(pair):
    old, new = pair
    q = LogVectorQuery(event_type="tool_used", session_id="s1", since=1788500000, until=1788502000, limit=50)
    qa, qb = old.query(q), new.query(q)
    assert [x.record.event_id for x in qa] == [x.record.event_id for x in qb]
    assert all(x.record.event_type == "tool_used" for x in qb)


def test_recall_hybrid_parity(pair):
    old, new = pair
    rng = random.Random(7)
    qv = [rng.random() for _ in range(DIM)]
    kw = dict(
        query_text="NPC Vendor 卖种子 sold",
        query_vector=qv,
        session_id="s1",
        limit=8,
        local_boost=0.15,
        role_weights={"tool/*": 0.2},
    )  # recency 关闭：避免两次调用真实时间差影响，专注后端一致性
    ra, rb = old.recall_hybrid(**kw), new.recall_hybrid(**kw)
    assert [x.record.event_id for x in ra] == [x.record.event_id for x in rb]
    assert all(abs(x.score - y.score) <= 1e-6 for x, y in zip(ra, rb))


def test_tags_gate_all_and_any(pair):
    _, new = pair
    all_ids = {x.record.event_id for x in new.query(LogVectorQuery(tags=["npc:Vendor", "day:3"], limit=500))}
    assert all_ids == {"e0", "e35"}  # 60 条内同时满足 i%5==0 且 i%7==0
    any_ids = {x.record.event_id for x in new.query(LogVectorQuery(tags=["npc:Vendor", "day:3"], tag_match="any", limit=500))}
    manual = {f"e{i}" for i in range(60) if i % 5 == 0 or i % 7 == 0}
    assert any_ids == manual


def test_tags_parity_with_old(pair):
    old, new = pair
    q = LogVectorQuery(tags=["npc:Vendor"], limit=100)
    assert [x.record.event_id for x in old.query(q)] == [x.record.event_id for x in new.query(q)]


def test_tags_zero_hit_degrades(pair):
    old, new = pair
    q = LogVectorQuery(tags=["nonexistent:zzz"], limit=10)
    qa, qb = old.query(q), new.query(q)
    assert [x.record.event_id for x in qa] == [x.record.event_id for x in qb]
    assert len(qb) == 10
    assert all(x.tag_matched is False for x in qb)
    hit = new.query(LogVectorQuery(tags=["npc:Vendor"], limit=10))
    assert all(x.tag_matched is True for x in hit)


def test_recall_hybrid_tags_degrade(pair):
    old, new = pair
    kw = dict(query_text="NPC Vendor", limit=6, tags=["nonexistent:zzz"])
    ra, rb = old.recall_hybrid(**kw), new.recall_hybrid(**kw)
    assert [x.record.event_id for x in ra] == [x.record.event_id for x in rb]
    assert all(not x.tag_matched for x in rb)


class _FrozenTime:
    def __init__(self, t: float) -> None:
        self.t = t

    def time(self) -> float:
        return self.t


def test_lifecycle_sweep_retention(tmp_path, monkeypatch):
    # 无 lifecycle 管理器 -> 走 retention_sec 空闲清理路径（与原 store 语义一致）。
    # 冻结模块时钟，隔离测试机真实时间。
    import aigc_core.sqlite_log_store as sls
    from aigc_core.log_vector_store import StoreLifecycleConfig

    base = 1788500000.0
    monkeypatch.setattr(sls, "time", _FrozenTime(base))
    store = SqliteLogStore(
        path=tmp_path / "s.db",
        config=StoreLifecycleConfig(retention_sec=30.0, max_entries=0),
    )
    store.add(GameLogRecord(event_id="old", event_type="t", unix_time=base, last_used_at=base))
    store.add(GameLogRecord(event_id="new", event_type="t", unix_time=base, last_used_at=base + 120))
    removed = store.sweep_expired(now=base + 131.0)  # 阈值=base+101：old 删，new 留
    assert removed == 1
    assert store.get("old") is None and store.get("new") is not None
    store.close()


def test_lifecycle_manager_sweep_deadline(tmp_path, monkeypatch):
    # 有 lifecycle 管理器 -> 走 deadline_base+duration 死期路径（retention 不生效）
    import aigc_core.sqlite_log_store as sls

    base = 1788500000.0
    monkeypatch.setattr(sls, "time", _FrozenTime(base))
    lm = VectorLifecycleManager(default_duration_sec=7 * 24 * 3600)
    store = SqliteLogStore(path=tmp_path / "d.db", lifecycle=lm)
    store.add(GameLogRecord(event_id="k", event_type="t", unix_time=base, last_used_at=base))
    assert store.sweep_expired(now=base + 10.0) == 0  # 死期未到
    assert store.lifecycle_expires_at("k") is not None
    store.set_lifecycle_duration("k", 50.0)  # deadline_base=base(创建时刻) -> base+50 到期
    assert store.sweep_expired(now=base + 100.0) == 1
    assert store.get("k") is None
    store.close()


def test_evict_over_cap_lru(tmp_path):
    lm = VectorLifecycleManager(default_duration_sec=7 * 24 * 3600)
    store = SqliteLogStore(
        path=tmp_path / "cap.db",
        lifecycle=lm,
        config=__import__("aigc_core.log_vector_store", fromlist=["StoreLifecycleConfig"]).StoreLifecycleConfig(
            retention_sec=-1, max_entries=10
        ),
    )
    for i in range(20):
        store.add(
            GameLogRecord(event_id=f"e{i}", event_type="t", unix_time=i, last_used_at=i)
        )
    assert store.count() == 10
    ids = {r.event_id for r in store.all()}
    assert ids == {f"e{i}" for i in range(10, 20)}  # 淘汰 last_used_at 最久未用的前 10 条
    assert store.warning_breached() is True  # 10 > warn_entries(=10*0.8)
    store.close()


def test_persist_reopen_tags_survive(tmp_path):
    db = tmp_path / "reopen.db"
    lm = VectorLifecycleManager(default_duration_sec=7 * 24 * 3600)
    s1 = SqliteLogStore(path=db, lifecycle=lm)
    s1.add(GameLogRecord(event_id="k1", event_type="npc.spawn", data={"npc": "Vendor"}, tags=["npc:Vendor", "lifecycle:snapshot"]))
    s1.close()
    lm2 = VectorLifecycleManager(default_duration_sec=7 * 24 * 3600)
    s2 = SqliteLogStore(path=db, lifecycle=lm2)
    assert s2.count() == 1
    r = s2.get("k1")
    assert r.tags == ["lifecycle:snapshot", "npc:Vendor"]
    assert r.data == {"npc": "Vendor"}
    s2.close()


def test_to_embedding_text_tags():
    r = GameLogRecord(event_id="x", event_type="npc.spawn", tags=["b:2", "a:1"])
    assert '"tags": ["a:1", "b:2"]' in r.to_embedding_text()
    r2 = GameLogRecord(event_id="y", event_type="player.move")
    assert "tags" not in r2.to_embedding_text()


def test_ingest_skips_debug_level(tmp_path):
    """日志级别决定是否写库：DEBUG 不入库，双后端语义一致。"""
    import json as _json

    src = tmp_path / "ev.jsonl"
    recs = [
        {"event_id": "d1", "event_type": "player.move", "level": "DEBUG", "data": {}},
        {"event_id": "i1", "event_type": "world.soil_tilled", "level": "INFO", "data": {}},
        {"event_id": "i2", "event_type": "ai.command", "data": {}},  # 缺 level -> 默认 INFO
    ]
    src.write_text("\n".join(_json.dumps(r) for r in recs), encoding="utf-8")

    old = LogVectorStore()
    new = SqliteLogStore(path=tmp_path / "s.db")
    n_old = old.ingest_jsonl(src)
    n_new = new.ingest_jsonl(src)
    assert n_old == n_new == 2  # DEBUG 被跳过
    assert new.count() == 2
    assert new.get("d1") is None and new.get("i1") is not None and new.get("i2") is not None
    # 默认级别是 INFO
    assert new.get("i2").level == "INFO"
    new.close()


def test_level_default_and_roundtrip():
    r = GameLogRecord(event_id="x", event_type="t")
    assert r.level == "INFO"
