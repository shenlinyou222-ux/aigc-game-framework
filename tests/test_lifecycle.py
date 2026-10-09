"""向量生命周期管理（死期+持续时间）与 LogVectorStore 集成测试。

覆盖：
- create 默认 duration / deadline_base → expires_at = base + duration
- touch 重置死期、不改 duration
- set_duration 改 duration、保留死期基准
- is_expired / sweep 判定与清理
- CRUD 策略注册表的热更换（hot-swap）
- LogVectorStore 传入 lifecycle 后 touch / set_duration / sweep 走新语义
"""

from __future__ import annotations

import time
import unittest

from fastapi.testclient import TestClient

from aigc_core.api import create_app
from aigc_core.controller import MainController
from aigc_core.lifecycle import (
    LifecycleMeta,
    LifecycleOp,
    LifecyclePolicyRegistry,
    LifecycleUpdateParams,
    SimpleLifecyclePolicy,
    VectorLifecycleManager,
)
from aigc_core.log_vector_store import LogVectorStore
from aigc_core.models import GameLogRecord


class LifecycleManagerTests(unittest.TestCase):
    def test_create_sets_default_duration_and_base(self) -> None:
        mgr = VectorLifecycleManager(default_duration_sec=120.0)
        mgr.create("e1", now=1000.0)
        meta = mgr.get("e1")
        assert meta is not None
        self.assertEqual(meta.deadline_base, 1000.0)
        self.assertEqual(meta.duration_sec, 120.0)
        self.assertEqual(meta.expires_at(), 1120.0)
        self.assertEqual(mgr.expires_at("e1"), 1120.0)

    def test_touch_resets_deadline_keeps_duration(self) -> None:
        mgr = VectorLifecycleManager(default_duration_sec=120.0)
        mgr.create("e1", now=1000.0)
        mgr.touch("e1", now=2000.0)
        meta = mgr.get("e1")
        assert meta is not None
        self.assertEqual(meta.deadline_base, 2000.0)   # 死期基准被重置
        self.assertEqual(meta.duration_sec, 120.0)      # duration 不被重置
        self.assertEqual(meta.expires_at(), 2120.0)

    def test_set_duration_changes_duration_keeps_base(self) -> None:
        mgr = VectorLifecycleManager(default_duration_sec=120.0)
        mgr.create("e1", now=1000.0)
        mgr.set_duration("e1", 3600.0)
        meta = mgr.get("e1")
        assert meta is not None
        self.assertEqual(meta.duration_sec, 3600.0)
        self.assertEqual(meta.deadline_base, 1000.0)   # base 未动
        self.assertEqual(meta.expires_at(), 4600.0)

    def test_invalid_duration_rejected(self) -> None:
        mgr = VectorLifecycleManager()
        mgr.create("e1", now=0.0)
        with self.assertRaises(ValueError):
            mgr.set_duration("e1", 0)

    def test_sweep_finds_expired_only(self) -> None:
        mgr = VectorLifecycleManager(default_duration_sec=100.0)
        mgr.create("old", now=0.0)     # expires_at=100
        mgr.create("new", now=1000.0)  # expires_at=1100
        expired = mgr.sweep(now=150.0)
        self.assertEqual(expired, ["old"])

    def test_is_expired_after_reset(self) -> None:
        mgr = VectorLifecycleManager(default_duration_sec=100.0)
        mgr.create("e1", now=0.0)
        self.assertTrue(mgr.is_expired("e1", now=101.0))
        mgr.touch("e1", now=500.0)     # 续命
        self.assertFalse(mgr.is_expired("e1", now=550.0))
        self.assertTrue(mgr.is_expired("e1", now=601.0))


class HotSwapPolicyTests(unittest.TestCase):
    def test_swap_update_policy_changes_behavior(self) -> None:
        """默认：touch 只重置 deadline。换一个“touch 同时把 duration 拉回默认”的策略。"""

        class ExtendPolicy(SimpleLifecyclePolicy):
            name = "extend-on-update"

            def __init__(self) -> None:
                super().__init__(default_duration_sec=600.0)

            def on_update(
                self,
                meta: LifecycleMeta,
                now: float,
                params: LifecycleUpdateParams | None = None,
            ) -> None:
                # 先按默认续命，再把 duration 重置回较长默认（模拟“读即延长”策略）
                super().on_update(
                    meta,
                    now,
                    LifecycleUpdateParams(reset_deadline=True, duration_sec=self.default_duration_sec),
                )

        registry = LifecyclePolicyRegistry(SimpleLifecyclePolicy(default_duration_sec=120.0))
        mgr = VectorLifecycleManager(registry=registry)
        mgr.create("e1", now=0.0)
        mgr.touch("e1", now=100.0)
        meta = mgr.get("e1")
        assert meta is not None
        self.assertEqual(meta.duration_sec, 120.0)  # 默认策略：duration 不变

        # 热更换 UPDATE 策略
        registry.swap(LifecycleOp.UPDATE, ExtendPolicy())
        mgr.touch("e1", now=200.0)
        meta2 = mgr.get("e1")
        assert meta2 is not None
        self.assertEqual(meta2.deadline_base, 200.0)
        self.assertEqual(meta2.duration_sec, 600.0)  # 新策略生效

    def test_registry_snapshot_reports_ops(self) -> None:
        registry = LifecyclePolicyRegistry(SimpleLifecyclePolicy())
        snap = registry.snapshot()
        self.assertEqual(set(snap), {op.value for op in LifecycleOp})


class LogVectorStoreLifecycleIntegrationTests(unittest.TestCase):
    def test_store_touch_and_set_duration_route_to_manager(self) -> None:
        mgr = VectorLifecycleManager(default_duration_sec=1800.0)
        store = LogVectorStore(lifecycle=mgr)
        store.add(GameLogRecord(event_id="e1", event_type="user/message", kind="user/message",
                                unix_time=time.time(), session_id="s1", data={"text": "hi"}))
        self.assertIsNotNone(store.lifecycle_expires_at("e1"))

        # 改 duration → 理论删除时刻顺延
        new_exp = store.set_lifecycle_duration("e1", 7200.0)
        self.assertIsNotNone(new_exp)
        meta = store.lifecycle_meta("e1")
        assert meta is not None
        self.assertEqual(meta["duration_sec"], 7200.0)

        # touch 续命：只改 deadline_base
        before = mgr.get("e1").expires_at()
        store.touch("e1", at=before + 1)
        after = mgr.get("e1").expires_at()
        self.assertGreater(after, before)

    def test_store_sweep_removes_deadline_expired(self) -> None:
        mgr = VectorLifecycleManager(default_duration_sec=100.0)
        store = LogVectorStore(lifecycle=mgr)
        store.add(GameLogRecord(event_id="dead", event_type="user/message", unix_time=0.0,
                                session_id="s1", data={"text": "old"}))
        # 把这条的死期基准推到很久以前，使其 expires_at=0+100=100
        mgr.get("dead").deadline_base = 0.0
        removed = store.sweep_expired(now=150.0)
        self.assertEqual(removed, 1)
        self.assertIsNone(store.get("dead"))

    def test_store_delete_cleans_manager_meta(self) -> None:
        mgr = VectorLifecycleManager()
        store = LogVectorStore(lifecycle=mgr)
        store.add(GameLogRecord(event_id="e1", event_type="user/message", unix_time=time.time(), data={}))
        self.assertEqual(mgr.count(), 1)
        store.delete("e1")
        self.assertEqual(mgr.count(), 0)


class LifecycleApiTests(unittest.TestCase):
    def _client(self) -> TestClient:
        ctrl = MainController(log_lifecycle=VectorLifecycleManager(default_duration_sec=120.0))
        return TestClient(create_app(controller=ctrl))

    def test_lifecycle_meta_and_duration(self) -> None:
        client = self._client()
        now = time.time()
        r = client.post(
            "/ctx/ingest",
            json={"record": {"event_id": "e1", "event_type": "user/message",
                             "kind": "user/message", "unix_time": now,
                             "session_id": "s1", "data": {"text": "hello"}}},
        )
        self.assertEqual(r.status_code, 200)
        self.assertEqual(r.json()["event_id"], "e1")

        status = client.get("/ctx/lifecycle/status").json()
        self.assertTrue(status["enabled"])

        meta = client.get("/ctx/lifecycle/meta/e1").json()
        self.assertTrue(meta["found"])
        self.assertEqual(meta["meta"]["duration_sec"], 120.0)

        # 显式修改持续时间
        d = client.post("/ctx/lifecycle/duration", json={"event_id": "e1", "duration_sec": 600.0})
        self.assertEqual(d.status_code, 200)
        self.assertTrue(d.json()["ok"])

        # 重置死期（续命）
        reset = client.post("/ctx/lifecycle/reset", json={"event_id": "e1"})
        self.assertEqual(reset.json(), {"ok": True, "event_id": "e1"})


if __name__ == "__main__":
    unittest.main()
