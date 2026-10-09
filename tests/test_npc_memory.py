# -*- coding: utf-8 -*-
"""AI-NPC 记忆口子测试：置信度门控召回 / 推理新向量 / CEL 权限规则。"""
import tempfile
import unittest
from pathlib import Path

from aigc_core.functions import FunctionRegistry, RuleError
from aigc_core.interface_store import HashEmbedder
from aigc_core.log_vector_store import LogVectorStore
from aigc_core.models import GameLogRecord
from aigc_core.npc_memory import (
    NpcMemory,
    SOURCE_CONFIDENCE,
    build_npc_prompt,
    confidence_of,
    load_personas,
)


def _personas() -> dict:
    return load_personas(Path(__file__).resolve().parents[1] / "npc_personas.json")


def _make(tmp: Path) -> tuple:
    store = LogVectorStore(path=tmp / "npc_store.jsonl")
    nm = NpcMemory(store, _personas(), embedder=HashEmbedder(), actor="system")
    return store, nm


def _add(store: LogVectorStore, eid: str, text: str, source_type: str,
         conf=None, session: str = "") -> None:
    data = {"text": text, "source_type": source_type}
    if conf is not None:
        data["confidence"] = conf
    store.add(GameLogRecord(
        event_id=eid, event_type="npc.dialogue_line", kind="event",
        session_id=session, unix_time=1000.0, data=data))


class TestSourceConfidence(unittest.TestCase):
    def test_template_matches_design(self):
        # 01§6.3 信源模板
        self.assertEqual(SOURCE_CONFIDENCE["亲眼"], 0.95)
        self.assertEqual(SOURCE_CONFIDENCE["亲耳"], 0.70)
        self.assertEqual(SOURCE_CONFIDENCE["同阵营转述"], 0.55)
        self.assertEqual(SOURCE_CONFIDENCE["敌对阵营转述"], 0.15)
        self.assertEqual(SOURCE_CONFIDENCE["系统公告"], 0.99)
        self.assertEqual(SOURCE_CONFIDENCE["推理"], 0.55)

    def test_confidence_of_prefers_explicit(self):
        rec = GameLogRecord(event_id="a", event_type="x", unix_time=1.0,
                            data={"source_type": "亲眼", "confidence": 0.42})
        self.assertEqual(confidence_of(rec), 0.42)
        rec2 = GameLogRecord(event_id="b", event_type="x", unix_time=1.0,
                             data={"source_type": "系统公告"})
        self.assertEqual(confidence_of(rec2), 0.99)
        rec3 = GameLogRecord(event_id="c", event_type="x", unix_time=1.0, data={})
        self.assertEqual(confidence_of(rec3), 0.5)


class TestNpcRecall(unittest.TestCase):
    def test_unknown_npc_rejected_by_cel(self):
        tmp = tempfile.mkdtemp()
        _, nm = _make(Path(tmp))
        with self.assertRaises(RuleError) as cm:
            nm.recall("dog", "玩家昨天做了什么")
        self.assertEqual(cm.exception.rule, "npc_known")

    def test_actor_permission_by_cel(self):
        tmp = tempfile.mkdtemp()
        _, nm = _make(Path(tmp))
        nm.actor = "mouse"  # 鼠商人想查猫的记忆
        with self.assertRaises(RuleError) as cm:
            nm.recall("cat", "玩家昨天做了什么")
        self.assertEqual(cm.exception.rule, "actor_allowed")

    def test_threshold_sanity_by_cel(self):
        tmp = tempfile.mkdtemp()
        store, _ = _make(Path(tmp))
        personas = _personas()
        personas["cat"]["memory"]["confidence_threshold"] = 1.5  # 非法
        nm = NpcMemory(store, personas, embedder=HashEmbedder(), actor="system")
        with self.assertRaises(RuleError) as cm:
            nm.recall("cat", "玩家昨天做了什么")
        self.assertEqual(cm.exception.rule, "threshold_sane")

    def test_recall_gate_filters_distrust_source(self):
        tmp = tempfile.mkdtemp()
        store, nm = _make(Path(tmp))
        _add(store, "h1", "玩家亲手给我烤了面包", "亲眼")
        _add(store, "h2", "听说玩家把田全荒了", "敌对阵营转述")
        r = nm.recall("cat", "玩家做过什么")
        ids = [it["event_id"] for it in r["items"]]
        self.assertIn("h1", ids)
        self.assertNotIn("h2", ids)
        self.assertGreaterEqual(r["dropped_by_gate"], 1)
        self.assertEqual(r["gate"], "cand_conf >= threshold")

    def test_recall_custom_gate_is_data(self):
        # gate 是人物数据里的 CEL 表达式：换成"只信亲眼"立即生效
        tmp = tempfile.mkdtemp()
        store, _ = _make(Path(tmp))
        _add(store, "h1", "玩家昨天砍了 3 棵树", "亲眼")
        _add(store, "h2", "公告：明天有雨", "系统公告")
        personas = _personas()
        personas["cat"]["memory"]["gate"] = "cand_source == '亲眼'"
        nm = NpcMemory(store, personas, embedder=HashEmbedder(), actor="system")
        r = nm.recall("cat", "昨天发生了什么")
        ids = [it["event_id"] for it in r["items"]]
        self.assertIn("h1", ids)
        self.assertNotIn("h2", ids)

    def test_touch_extends_life(self):
        tmp = tempfile.mkdtemp()
        store, nm = _make(Path(tmp))
        rec = GameLogRecord(event_id="t1", event_type="npc.dialogue_line", kind="event",
                            unix_time=1000.0,
                            data={"text": "玩家送过我一条鱼", "source_type": "亲眼",
                                  "confidence": 0.9})
        if store.lifecycle is not None:
            store.add(rec)
            before = store.lifecycle_meta("t1")
            nm.recall("cat", "送鱼")
            after = store.lifecycle_meta("t1")
            self.assertGreaterEqual(after.deadline_base, before.deadline_base)
        else:
            store.add(rec)
            r = nm.recall("cat", "送鱼")
            self.assertIn("t1", [it["event_id"] for it in r["items"]])


class TestInferenceStore(unittest.TestCase):
    def test_self_write_only_by_cel(self):
        tmp = tempfile.mkdtemp()
        _, nm = _make(Path(tmp))
        nm.actor = "mouse"  # 别的 NPC 想替猫写记忆
        with self.assertRaises(RuleError) as cm:
            nm.store_inference("cat", "玩家似乎喜欢面包")
        self.assertEqual(cm.exception.rule, "self_write_only")

    def test_short_text_rejected_by_cel(self):
        tmp = tempfile.mkdtemp()
        _, nm = _make(Path(tmp))
        nm.actor = "cat"
        with self.assertRaises(RuleError) as cm:
            nm.store_inference("cat", "嗯")
        self.assertEqual(cm.exception.rule, "text_min")

    def test_gate_blocks_ungrounded_high_claim(self):
        tmp = tempfile.mkdtemp()
        store, nm = _make(Path(tmp))
        nm.actor = "cat"
        r = nm.store_inference("cat", "玩家其实很有钱", grounded=False, confidence=0.9)
        self.assertFalse(r["stored"])
        self.assertEqual(r["reason"], "gate")

    def test_store_success_writes_to_own_session(self):
        tmp = tempfile.mkdtemp()
        store, nm = _make(Path(tmp))
        nm.actor = "cat"
        r = nm.store_inference(
            "cat", "玩家连着三天都来喂我",
            grounded=True, confidence=0.8,
            claims=[{"text": "玩家三天都来喂我", "event_type": "npc.dialogue_line", "day": 3}])
        self.assertTrue(r["stored"])
        self.assertEqual(r["session"], "npc/cat")
        hits = store.recall_hybrid(query_text="玩家 喂", session_id="npc/cat", limit=5)
        self.assertTrue(any(h.record.event_id == r["event_id"] for h in hits))
        self.assertEqual(hits[0].record.data.get("source_type"), "推理")

    def test_registry_call(self):
        tmp = tempfile.mkdtemp()
        store, nm = _make(Path(tmp))
        reg = nm.register(FunctionRegistry())
        out = reg.call("npc_recall", {"npc_id": "cat", "query_text": "玩家是谁"},
                       nm._env("cat"))
        self.assertIn("memory_block", out)


class TestPrompt(unittest.TestCase):
    def test_prompt_has_persona_memory_and_claims(self):
        personas = _personas()
        prompt = build_npc_prompt(personas["cat"],
                                  "- [亲眼|置信0.95] 玩家砍了树", "你好呀")
        self.assertIn("傲娇猫", prompt)
        self.assertIn("哼", prompt)
        self.assertIn("玩家砍了树", prompt)
        self.assertIn("claims", prompt)
        self.assertIn("CONTRADICTED", prompt)

    def test_prompt_empty_memory(self):
        prompt = build_npc_prompt({"display_name": "测试NPC", "speech": {}},
                                  "", "嗨")
        self.assertIn("暂无", prompt)


if __name__ == "__main__":
    unittest.main()
