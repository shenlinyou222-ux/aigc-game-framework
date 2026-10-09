"""日志向量库测试：模型解析、向量召回、过滤、JSONL 持久化。"""

from __future__ import annotations

import json
import tempfile
import time
import unittest
from pathlib import Path

from aigc_core import (
    GameLogRecord,
    LogVectorQuery,
    LogVectorStore,
    MainController,
    StoreLifecycleConfig,
)
from aigc_core.models import LogSearchResult


class GameLogRecordTests(unittest.TestCase):
    def test_parse_godot_jsonl_shape(self) -> None:
        raw = {
            "event_id": "player.move.12345",
            "event_type": "player.move",
            "frame": 123,
            "time": "2026-09-02T19:22:43",
            "unix_time": 1788348163.311,
            "data": {"direction": [1.0, 0.0], "moving": True},
        }
        record = GameLogRecord.model_validate(raw)
        self.assertEqual(record.event_type, "player.move")
        self.assertEqual(record.frame, 123)
        self.assertEqual(record.data["moving"], True)
        self.assertEqual(record.semantic_vector, [])

    def test_to_embedding_text_contains_event_and_data(self) -> None:
        record = GameLogRecord(
            event_id="npc.1",
            event_type="npc.dialogue_line",
            data={"text": "你好"},
        )
        text = record.to_embedding_text()
        self.assertIn("npc.dialogue_line", text)
        self.assertIn("你好", text)


class LogVectorStoreTests(unittest.TestCase):
    def setUp(self) -> None:
        self.store = LogVectorStore()
        self.store.add(
            GameLogRecord(
                event_id="e1",
                event_type="player.move",
                unix_time=100.0,
                semantic_vector=[1.0, 0.0, 0.0],
            )
        )
        self.store.add(
            GameLogRecord(
                event_id="e2",
                event_type="npc.dialogue_line",
                unix_time=200.0,
                semantic_vector=[0.0, 1.0, 0.0],
            )
        )

    def test_vector_query_ranks_similar_higher(self) -> None:
        results = self.store.query(
            LogVectorQuery(query_vector=[0.9, 0.1, 0.0], limit=10)
        )
        self.assertEqual(len(results), 2)
        self.assertEqual(results[0].record.event_id, "e1")
        self.assertGreater(results[0].score, results[1].score)

    def test_empty_vector_query_filters_recent(self) -> None:
        results = self.store.query(LogVectorQuery(event_type="npc.dialogue_line"))
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].record.event_id, "e2")

    def test_time_window_filter(self) -> None:
        results = self.store.query(LogVectorQuery(since=150.0, until=250.0))
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].record.event_id, "e2")

    def test_query_min_score(self) -> None:
        results = self.store.query(
            LogVectorQuery(query_vector=[1.0, 0.0, 0.0], min_score=0.99)
        )
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].record.event_id, "e1")

    def test_persist_and_reload(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "log_vector_store.jsonl"
            store = LogVectorStore(path=path)
            store.add(
                GameLogRecord(
                    event_id="persist-1",
                    event_type="shop.purchase",
                    unix_time=1.0,
                    data={"item": "seed"},
                    semantic_vector=[0.2, 0.8],
                )
            )

            reloaded = LogVectorStore(path=path)
            self.assertEqual(reloaded.count(), 1)
            record = reloaded.get("persist-1")
            self.assertIsNotNone(record)
            self.assertEqual(record.event_type, "shop.purchase")
            self.assertEqual(record.semantic_vector, [0.2, 0.8])

    def test_ingest_jsonl_from_godot_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp) / "game_events.jsonl"
            lines = [
                {
                    "event_id": "move.1",
                    "event_type": "player.move",
                    "time": "t",
                    "unix_time": 1.0,
                    "frame": 1,
                    "data": {"moving": True},
                },
                {
                    "event_id": "npc.1",
                    "event_type": "npc.dialogue_line",
                    "time": "t",
                    "unix_time": 2.0,
                    "frame": 2,
                    "data": {"text": "你好"},
                },
            ]
            source.write_text(
                "\n".join(json.dumps(line, ensure_ascii=False) for line in lines) + "\n",
                encoding="utf-8",
            )

            store = LogVectorStore()

            def embedder(record: GameLogRecord):
                if "npc" in record.event_type:
                    return [0.0, 1.0]
                return [1.0, 0.0]

            count = store.ingest_jsonl(source, embedder=embedder)
            self.assertEqual(count, 2)
            self.assertEqual(store.get("move.1").semantic_vector, [1.0, 0.0])
            self.assertEqual(store.get("npc.1").semantic_vector, [0.0, 1.0])

            results = store.query(LogVectorQuery(query_vector=[0.0, 1.0]))
            self.assertEqual(results[0].record.event_id, "npc.1")


class ControllerLogTests(unittest.TestCase):
    def test_controller_record_and_query_log(self) -> None:
        ctrl = MainController()
        ctrl.record_log(
            GameLogRecord(
                event_id="log-1",
                event_type="player.tool_used",
                unix_time=1.0,
                semantic_vector=[1.0, 0.0],
            )
        )
        result = ctrl.query_logs(
            LogVectorQuery(query_vector=[0.9, 0.0], event_type="player.tool_used")
        )
        self.assertIsInstance(result[0], LogSearchResult)
        self.assertEqual(result[0].record.event_id, "log-1")
        self.assertIsNotNone(ctrl.get_log("log-1"))


class LifecycleTests(unittest.TestCase):
    """生命周期管理测试：空闲超时删除、touch 续命、条数上限淘汰、警戒线、持久化。"""

    def _config(self, retention_sec: float = 60.0, max_entries: int = 100) -> StoreLifecycleConfig:
        return StoreLifecycleConfig(
            retention_sec=retention_sec,
            max_entries=max_entries,
            warn_ratio=0.5,
        )

    def _record(
        self,
        event_id: str,
        last_used_at: float,
        unix_time: Optional[float] = None,
    ) -> GameLogRecord:
        return GameLogRecord(
            event_id=event_id,
            event_type="player.move",
            unix_time=time.time() if unix_time is None else unix_time,
            last_used_at=last_used_at,
        )

    def test_add_initializes_last_used_at_when_missing(self) -> None:
        store = LogVectorStore(config=self._config())
        before = time.time()
        store.add(GameLogRecord(event_id="fresh", event_type="player.move", unix_time=1.0))
        after = time.time()
        record = store.get("fresh")
        self.assertGreaterEqual(record.last_used_at, before)
        self.assertLessEqual(record.last_used_at, after)

    def test_sweep_expired_removes_idle_entries_only(self) -> None:
        store = LogVectorStore(config=self._config(retention_sec=100.0))
        base = time.time()
        store.add(self._record("old", last_used_at=base))        # 在 base+200 时已空闲 200
        store.add(self._record("recent", last_used_at=base + 150))  # 空闲仅 50 -> 留

        removed = store.sweep_expired(now=base + 200)
        self.assertEqual(removed, 1)
        self.assertIsNone(store.get("old"))
        self.assertIsNotNone(store.get("recent"))

    def test_touch_resets_idle_timer_and_keeps_unix_time(self) -> None:
        store = LogVectorStore(config=self._config(retention_sec=100.0))
        base = time.time()
        store.add(self._record("e", last_used_at=base, unix_time=777.0))  # 初始空闲基准 base

        self.assertTrue(store.touch("e", at=base + 200))  # 注入上下文前续命
        # 续命后（基准 base+200），在 base+250 时空闲仅 50 -> 不删
        self.assertEqual(store.sweep_expired(now=base + 250), 0)
        # 原始 unix_time 不应被 touch 覆盖
        self.assertEqual(store.get("e").unix_time, 777.0)
        # 续命后再空闲 150（到 base+350）才删
        self.assertEqual(store.sweep_expired(now=base + 350), 1)
        self.assertIsNone(store.get("e"))

    def test_evict_over_cap_keeps_newest_used(self) -> None:
        store = LogVectorStore(config=self._config(retention_sec=0.0, max_entries=3))
        store.add(self._record("a", last_used_at=1.0))
        store.add(self._record("b", last_used_at=2.0))
        store.add(self._record("c", last_used_at=3.0))
        store.add(self._record("d", last_used_at=4.0))  # 触发淘汰最旧 a

        self.assertEqual(store.count(), 3)
        self.assertIsNone(store.get("a"))
        for keep in ("b", "c", "d"):
            self.assertIsNotNone(store.get(keep))

    def test_warning_breached_over_warn_entries(self) -> None:
        store = LogVectorStore(config=self._config(retention_sec=0.0, max_entries=10))
        self.assertFalse(store.warning_breached())
        for i in range(6):  # warn_entries = int(10*0.5) = 5
            store.add(self._record(f"w{i}", last_used_at=float(i)))
        self.assertTrue(store.warning_breached())
        status = store.capacity_status()
        self.assertEqual(status["count"], 6)
        self.assertTrue(status["over_warning"])

    def test_lazy_sweep_on_query(self) -> None:
        store = LogVectorStore(config=self._config(retention_sec=100.0))
        base = time.time()
        store.add(self._record("e", last_used_at=base))  # 先正常加入（不会被 add 清掉）
        # 模拟该记录在库里随时间老化到过期
        store._entries["e"].last_used_at = base - 200
        results = store.query(LogVectorQuery(limit=10))
        self.assertEqual(results, [])  # query 触发懒清理后已不在
        self.assertEqual(store.count(), 0)

    def test_last_used_at_persists_across_reload(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "log_vector_store.jsonl"
            store = LogVectorStore(path=path, config=self._config(retention_sec=100.0))
            base = time.time()
            store.add(self._record("p", last_used_at=base))

            reloaded = LogVectorStore(path=path, config=self._config(retention_sec=100.0))
            self.assertEqual(reloaded.get("p").last_used_at, base)

    def test_controller_touch_and_cleanup(self) -> None:
        store = LogVectorStore(config=self._config(retention_sec=100.0))
        ctrl = MainController(log_store=store)
        base = time.time()
        ctrl.record_log(self._record("c1", last_used_at=base))
        self.assertTrue(ctrl.touch_log("c1", at=base + 200))
        self.assertEqual(ctrl.get_log("c1").last_used_at, base + 200)
        result = ctrl.cleanup_logs()
        self.assertIn("swept_expired", result)
        self.assertIn("evicted_over_cap", result)
        # 续命后距真实 now 仍不足 retention，不应被清理
        self.assertEqual(result["swept_expired"], 0)
        self.assertIsNotNone(ctrl.get_log("c1"))


if __name__ == "__main__":
    unittest.main()
