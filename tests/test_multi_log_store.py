# -*- coding: utf-8 -*-
"""MultiLogStore: 路由 / 三库扇出统一打分 parity / milestone 永不删。"""
import random

import pytest

from aigc_core.log_vector_store import LogVectorStore
from aigc_core.models import GameLogRecord, LogVectorQuery
from aigc_core.multi_log_store import MultiLogStore
from aigc_core.sqlite_log_store import SqliteLogStore

DIM = 8
TEXTS = ["开垦土地", "白菜成熟", "NPC Vendor 卖种子", "敌人被击杀", "砍树得木头"]


def _mk(i: int) -> GameLogRecord:
    rng = random.Random(500 + i)
    et = ["world.soil_tilled", "world.plant_planted", "npc.dialogue_line",
          "world.enemy_died", "world.tree_hit"][i % 5]
    lvl = "MILESTONE" if et == "world.plant_planted" else "INFO"
    return GameLogRecord(
        event_id=f"m{i}", event_type=et, unix_time=1788500000 + i * 60,
        data={"text": TEXTS[i % 5], "n": i},
        semantic_vector=[rng.random() for _ in range(DIM)] if i % 2 == 0 else [],
        last_used_at=1788500000 + i, session_id=("s1" if i % 3 else ""),
        tags=(["npc:Vendor"] if et == "npc.dialogue_line" else []),
    )


@pytest.fixture()
def pair(tmp_path):
    import aigc_core.log_vector_store as lvs
    import aigc_core.sqlite_log_store as sls
    import aigc_core.multi_log_store as mls

    class FT:
        def __init__(self, t): self.t = t
        def time(self): return self.t

    frozen = FT(1788500400.0)
    origs = (lvs.time, sls.time, mls.time)
    lvs.time = sls.time = mls.time = frozen
    recs = [_mk(i) for i in range(40)]
    single = LogVectorStore()
    multi = MultiLogStore(tmp_path / "stores")
    for r in recs:
        single.add(r)
        multi.add(r)
    yield single, multi
    multi.close()
    lvs.time, sls.time, mls.time = origs


def test_routing_by_level_and_type(tmp_path):
    m = MultiLogStore(tmp_path / "r")
    m.add(GameLogRecord(event_id="ms", event_type="world.snapshot", level="MILESTONE"))
    m.add(GameLogRecord(event_id="dg", event_type="npc.dialogue_line"))
    m.add(GameLogRecord(event_id="ac", event_type="world.soil_tilled"))
    assert m.milestone.get("ms") is not None
    assert m.dialogue.get("dg") is not None
    assert m.activity.get("ac") is not None
    assert m.milestone.count() == 1 and m.dialogue.count() == 1 and m.activity.count() == 1
    m.close()


def test_recall_parity_single_vs_multi(pair):
    single, multi = pair
    rng = random.Random(9)
    qv = [rng.random() for _ in range(DIM)]
    kw = dict(query_text="白菜 Vendor 开垦", query_vector=qv, limit=10)
    ra = single.recall_hybrid(**kw)
    rb = multi.recall_hybrid(**kw)
    assert [x.record.event_id for x in ra] == [x.record.event_id for x in rb]
    assert all(abs(x.score - y.score) <= 1e-6 for x, y in zip(ra, rb))


def test_tags_gate_across_stores(pair):
    _, multi = pair
    hits = multi.query(LogVectorQuery(tags=["npc:Vendor"], limit=100))
    assert len(hits) == 8  # dialogue 通道 8 条全部带 npc:Vendor
    assert all(h.record.event_type == "npc.dialogue_line" for h in hits)


def test_milestone_store_never_sweeps(tmp_path):
    from aigc_core.multi_log_store import MILESTONE_CONFIG

    assert MILESTONE_CONFIG.retention_sec == -1 and MILESTONE_CONFIG.max_entries == -1
    m = MultiLogStore(tmp_path / "ms")
    m.add(GameLogRecord(event_id="ancient", event_type="world.snapshot",
                        level="MILESTONE", unix_time=1.0, last_used_at=1.0))
    assert m.sweep_expired(now=1788500000.0) == 0  # milestone 库无清理
    assert m.get("ancient") is not None
    m.close()


def test_token_cache_populates_and_matches(pair):
    from aigc_core import log_vector_store as lvs

    single, _ = pair
    r = single.get("m2")
    toks = lvs.record_tokens(r)
    assert toks == lvs._tokenize_text(r.to_embedding_text())
    assert "m2" in lvs._TOKEN_CACHE


def test_query_embed_cache():
    from aigc_core.embedders import QueryEmbedCache

    calls = []

    class Fake:
        def embed_text(self, text):
            calls.append(text)
            return [0.1, 0.2]

    c = QueryEmbedCache(Fake())
    assert c.embed("q") == [0.1, 0.2]
    assert c.embed("q") == [0.1, 0.2]
    assert len(c) == 1
    assert len(calls) == 1  # 第二次命中缓存


def test_world_baseline_exact_query(tmp_path):
    m = MultiLogStore(tmp_path / "wb")
    m.add(GameLogRecord(event_id="old", event_type="world.snapshot", level="MILESTONE",
                        unix_time=100.0, data={"day": 1}))
    m.add(GameLogRecord(event_id="new", event_type="world.snapshot", level="MILESTONE",
                        unix_time=200.0, data={"day": 2, "npcs": [{"npc": "Cat"}]}))
    base = m.world_baseline()
    assert base.event_id == "new"  # 最新快照
    assert base.data["npcs"] == [{"npc": "Cat"}]
    m.close()


def test_render_context_lossless(tmp_path):
    from aigc_core.render_context import render_context

    m = MultiLogStore(tmp_path / "rc")
    m.add(GameLogRecord(event_id="e1", event_type="world.soil_tilled",
                        data={"coord": [3, 12], "watered": False}, tags=["day:3"]))
    rec = m.get("e1")
    text = render_context([rec])
    import json as _json
    parsed = _json.loads(text)
    assert parsed[0]["event_type"] == "world.soil_tilled"
    assert parsed[0]["data"] == {"coord": [3, 12], "watered": False}
    assert parsed[0]["tags"] == ["day:3"]
    assert "semantic_vector" not in parsed[0]  # 向量永不入 prompt
    m.close()


def test_fuse_prefilter_topk():
    from aigc_core.log_vector_store import fuse_hybrid

    rng = random.Random(1)
    recs = []
    for i in range(600):
        recs.append(GameLogRecord(
            event_id=f"p{i}", event_type="world.x", unix_time=i,
            data={"t": f"event {i}"},
            semantic_vector=[rng.random() for _ in range(4)],
        ))
    qv = [rng.random() for _ in range(4)]
    out = fuse_hybrid(recs, now=1788500000.0, query_text="event",
                      query_vector=qv, limit=10, prefilter_top=100)
    assert len(out) == 10
    assert all(h.record.semantic_vector for h in out)  # 预筛后全是带向量的
