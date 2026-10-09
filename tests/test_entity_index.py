# -*- coding: utf-8 -*-
"""实体层（entity_index）测试：抽取规则 / 倒排持久化 / 召回加成 / 两后端 parity。"""
import pytest

from aigc_core.entity_index import EntityIndex, extract_entities, entity_raw_hits
from aigc_core.log_vector_store import LogVectorStore
from aigc_core.models import GameLogRecord
from aigc_core.sqlite_log_store import SqliteLogStore


# ---------- 抽取规则 ----------

def test_extract_windows_path():
    ents = extract_entities(r"改 E:\ai-repos\mem0\mem0\utils\entity_extraction.py 的规则")
    types = [t for t, _ in ents]
    assert "PATH" in types
    # 完整路径被抽成一条 PATH（不被 FILE/IDENTIFIER 拆走）
    path_ents = [tx for t, tx in ents if t == "PATH"]
    assert any("ai-repos" in p and "entity_extraction.py" in p for p in path_ents)


def test_extract_relative_path():
    ents = extract_entities("仓库在 repo\\src\\main.py 里")
    assert any(t == "PATH" and "main.py" in tx for t, tx in ents)


def test_extract_standalone_file():
    ents = extract_entities("看下 config.yml 的写法，以及 main.py")
    texts = [tx for t, tx in ents if t == "FILE"]
    assert "config.yml" in texts
    assert "main.py" in texts


def test_extract_identifiers():
    ents = extract_entities("ctx_recall 和 SqliteLogStore 是标识符；mem0.utils.entity_extraction 是模块链")
    texts = [tx for t, tx in ents]
    assert "ctx_recall" in texts
    assert "SqliteLogStore" in texts
    assert "mem0.utils.entity_extraction" in texts


def test_extract_quoted():
    ents = extract_entities('他说「直接切到 mem0」再谈')
    assert ("QUOTED", "直接切到 mem0") in ents


def test_generic_terms_filtered():
    # 英文/中文通用描述词不该成为实体
    ents = extract_entities("这个方法的结果是 task 的 type，整体情况没问题")
    assert ents == []


def test_entity_normalization_case_insensitive():
    idx = EntityIndex()
    idx.link("看 C:\\A\\B.py 的内容", "m1")
    boosts = idx.boosts_for([("PATH", "c:\\a\\b.py")])
    assert boosts.get("m1")


# ---------- 索引持久化 ----------

def test_index_persist_roundtrip(tmp_path):
    p = tmp_path / "entities.json"
    idx = EntityIndex(path=p)
    idx.link("改 E:\\a\\b.py 和 config.yml", "m1")
    idx.link("E:\\a\\b.py 已经改完", "m2")
    idx.flush()
    assert p.exists()

    idx2 = EntityIndex(path=p)
    assert idx2.count >= 2
    boosts = idx2.boosts_for([("PATH", "E:\\a\\b.py")])
    assert set(boosts.keys()) == {"m1", "m2"}
    assert boosts["m1"] == {"E:\\a\\b.py"}


def test_unlink_removes_memory(tmp_path):
    p = tmp_path / "e.json"
    idx = EntityIndex(path=p)
    idx.link("E:\\a\\b.py 的内容", "m1")
    idx.link("E:\\a\\b.py 的用法", "m2")
    removed = idx.unlink("m1")
    assert removed >= 1
    boosts = idx.boosts_for([("PATH", "E:\\a\\b.py")])
    assert "m1" not in boosts
    assert "m2" in boosts


def test_entity_raw_hits_empty_without_index():
    assert entity_raw_hits(None, "E:\\a\\b.py", ["m1"]) == {}
    assert entity_raw_hits(EntityIndex(), "", ["m1"]) == {}


# ---------- 召回加成 ----------

def _mk(event_id: str, text: str, ts: float) -> GameLogRecord:
    return GameLogRecord(
        event_id=event_id,
        event_type="user/message",
        unix_time=ts,
        data={"text": text},
    )


def test_recall_entity_boost_raises_hit():
    idx = EntityIndex()
    store = LogVectorStore()
    r1 = _mk("r1", "在 E:\\ai-repos\\mem0 里看 entity_extraction.py 的实现", 100.0)
    r2 = _mk("r2", "今天天气不错，去公园散步", 101.0)
    store.add(r1)
    store.add(r2)
    idx.link(r1.data["text"], "r1")

    base = store.recall_hybrid(query_text="E:\\ai-repos\\mem0 的实现", limit=10)
    boosted = store.recall_hybrid(
        query_text="E:\\ai-repos\\mem0 的实现",
        limit=10,
        entity_index=idx,
        entity_boost_weight=0.5,
    )

    b1 = next(h for h in boosted if h.record.event_id == "r1")
    assert b1.entity_hits, "实体命中应为非空"
    assert b1.entity_score > 0.0
    # 无实体层时子分为 0（向后兼容）
    a1 = next(h for h in base if h.record.event_id == "r1")
    assert a1.entity_hits == []
    assert a1.entity_score == 0.0
    # 实体加成后同一条记录总分不降
    assert b1.score >= a1.score


def test_recall_entity_boost_default_off():
    idx = EntityIndex()
    store = LogVectorStore()
    r1 = _mk("r1", "在 E:\\ai-repos\\mem0 里看 entity_extraction.py", 100.0)
    store.add(r1)
    idx.link(r1.data["text"], "r1")
    # 不传 entity_index：行为与旧版一致（无实体子分）
    hits = store.recall_hybrid(query_text="E:\\ai-repos\\mem0", limit=10)
    assert all(h.entity_hits == [] and h.entity_score == 0.0 for h in hits)


# ---------- 两后端 parity ----------

def test_entity_parity_jsonl_vs_sqlite(tmp_path):
    idx = EntityIndex()
    lvs = LogVectorStore()
    sls = SqliteLogStore(path=tmp_path / "parity.db")
    recs = [
        _mk("r1", "在 E:\\ai-repos\\mem0 里看 entity_extraction.py 的实现", 100.0),
        _mk("r2", "今天天气不错，去公园散步", 101.0),
        _mk("r3", "config.yml 的写法参考 main.py", 102.0),
    ]
    for r in recs:
        lvs.add(r)
        sls.add(r)
        idx.link(r.data["text"], r.event_id)

    kw = dict(
        query_text="E:\\ai-repos\\mem0 entity_extraction.py",
        limit=10,
        entity_index=idx,
        entity_boost_weight=0.5,
    )
    ra = lvs.recall_hybrid(**kw)
    rb = sls.recall_hybrid(**kw)
    assert [x.record.event_id for x in ra] == [x.record.event_id for x in rb]
    for x, y in zip(ra, rb):
        assert abs(x.score - y.score) <= 1e-9
        assert abs(x.entity_score - y.entity_score) <= 1e-9
        assert x.entity_hits == y.entity_hits


def test_recall_works_without_entities_regression():
    """实体层新参数全默认时，旧用例语义不变。"""
    store = LogVectorStore()
    store.add(_mk("r1", "白菜成熟了", 100.0))
    hits = store.recall_hybrid(query_text="白菜", limit=10)
    assert hits and hits[0].record.event_id == "r1"
    assert hits[0].entity_hits == []
