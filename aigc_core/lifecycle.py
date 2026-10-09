"""生命周期**策略→元数据翻译层**：死期(deadline_base) + 持续时间(duration)。

⚠️ 分层注意（重要）：
- **真正的向量库/数据库拥有数据的持久化与到期清理**，应用层不做内存对象簿记/遍历删。
- 本模块的职责只是把生命周期策略翻译成 **`expires_at = deadline_base + duration_sec`**
  并写进记录元数据（供向量库 `where=expires_at<now` 删除或原生 TTL）。
- 这里的 `VectorLifecycleManager`（内存 meta + `sweep`）**只服务于 JSONL 离线 demo
  （`LogVectorStore`）**，不是生产路径；生产用 `vector_db.py` 让 DB 删。
- 模型采用：
  - duration_sec  ：持续时间/存活时长。一般**不被重置**（续命只往后推死期基准），
                    但可以被显式修改（如把一条记忆从“短期”改成“长期”）。
  - deadline_base ：死期基准。`deadline_base + duration_sec` = 理论上应该被删除的时间。
                    向量被使用时可被 **重置为 now**（续命），duration 不变。
- CRUD（Create/Read/Update/Delete）四种操作各有策略；默认最简单策略，
  通过 Registry 支持**后续热更换策略**，DB 层无需改动。

本文件不依赖 pydantic / 向量库，可独立单测。
"""

from __future__ import annotations

import threading
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field, replace
from enum import Enum
from typing import Any, Dict, List, Optional


class LifecycleOp(str, Enum):
    """向量生命周期参与方会遇到的四种 CRUD 操作。"""

    CREATE = "create"
    READ = "read"
    UPDATE = "update"
    DELETE = "delete"


@dataclass
class LifecycleMeta:
    """一条上下文向量的生命周期元数据（与向量记录本身解耦）。

    - deadline_base + duration_sec 决定过期时刻 expires_at。
    - 当该向量被使用(注入上下文)时调用方应 reset_deadline（=续命），
      只改 deadline_base、不改 duration_sec。
    """

    event_id: str
    created_at: float = field(default_factory=time.time)
    deadline_base: float = field(default_factory=time.time)
    duration_sec: float = 1800.0          # 默认持续 30 分钟
    state: str = "alive"                  # alive | frozen | deleted

    def expires_at(self) -> float:
        return self.deadline_base + self.duration_sec

    def is_expired(self, now: Optional[float] = None) -> bool:
        if self.state == "deleted":
            return True
        now = time.time() if now is None else float(now)
        return now >= self.expires_at()

    def to_dict(self) -> Dict[str, Any]:
        return {
            "event_id": self.event_id,
            "created_at": round(self.created_at, 6),
            "deadline_base": round(self.deadline_base, 6),
            "duration_sec": round(self.duration_sec, 6),
            "expires_at": round(self.expires_at(), 6),
            "state": self.state,
        }


@dataclass
class LifecycleUpdateParams:
    """Update 操作可能携带的修改项。"""

    duration_sec: Optional[float] = None     # 显式改“持续时间”（一般不发生）
    reset_deadline: bool = False             # 重置“死期基准”=续命（可频繁发生）
    state: Optional[str] = None              # 冻结/删除等状态迁移


class VectorLifecyclePolicy(ABC):
    """单一 CRUD 操作的生命周期策略。每个策略只回答一个问题：
    当某条向量发生 Create/Read/Update/Delete 时，它的生命周期该怎么变。
    """

    name: str = "base"

    @abstractmethod
    def on_create(
        self,
        meta: LifecycleMeta,
        now: float,
        params: Optional[Dict[str, Any]] = None,
    ) -> LifecycleMeta:
        """决定一条新向量出生时的 deadline_base / duration_sec。"""

    @abstractmethod
    def on_read(self, meta: LifecycleMeta, now: float) -> None:
        """读命中时是否影响生命周期（默认策略：不续命，纯读）。"""

    @abstractmethod
    def on_update(
        self,
        meta: LifecycleMeta,
        now: float,
        params: Optional[LifecycleUpdateParams] = None,
    ) -> None:
        """Update 时的生命周期变化（改 duration / 重置 deadline / 改状态）。"""

    @abstractmethod
    def on_delete(self, meta: LifecycleMeta, now: float) -> None:
        """Delete 时标记/确认可删（默认策略直接置 deleted）。"""

    @abstractmethod
    def is_expired(self, meta: LifecycleMeta, now: float) -> bool:
        """判定此刻该向量是否过期（应由哪个策略负责清理）。"""


class SimpleLifecyclePolicy(VectorLifecyclePolicy):
    """当前选用的最简单策略：
    - Create ：duration = 默认值，deadline_base = now（立刻开始计时）
    - Read   ：纯读，不改变生命周期（需要“读即续命”请显式调 update(reset_deadline=True)）
    - Update ：显式改 duration 保留 deadline_base；reset_deadline 把 base 置 now；
               state 可迁移（如冻结）。
    - Delete ：置 deleted（过期即视为应删）。
    - is_expired：now >= deadline_base + duration_sec
    """

    name = "simple-expiry"

    def __init__(self, default_duration_sec: float = 1800.0) -> None:
        self.default_duration_sec = float(default_duration_sec)

    def on_create(
        self,
        meta: LifecycleMeta,
        now: float,
        params: Optional[Dict[str, Any]] = None,
    ) -> LifecycleMeta:
        params = params or {}
        duration = float(params.get("duration_sec", self.default_duration_sec))
        return replace(
            meta,
            created_at=now,
            deadline_base=now,
            duration_sec=duration,
            state="alive",
        )

    def on_read(self, meta: LifecycleMeta, now: float) -> None:
        # 纯读默认不续命，避免每次 query 都刷新导致“永远不死”。
        return None

    def on_update(
        self,
        meta: LifecycleMeta,
        now: float,
        params: Optional[LifecycleUpdateParams] = None,
    ) -> None:
        params = params or LifecycleUpdateParams()
        if params.reset_deadline:
            # 续命：只把死期基准推到 now，duration 保持不变。
            meta.deadline_base = now
        if params.duration_sec is not None:
            if params.duration_sec <= 0:
                raise ValueError("duration_sec must be > 0")
            meta.duration_sec = float(params.duration_sec)
            # 显式改 duration 不重置 base，让过期时刻按新 duration 顺延/缩短。
        if params.state is not None:
            meta.state = params.state

    def on_delete(self, meta: LifecycleMeta, now: float) -> None:
        meta.state = "deleted"

    def is_expired(self, meta: LifecycleMeta, now: float) -> bool:
        return meta.is_expired(now)


class LifecyclePolicyRegistry:
    """按 CRUD 操作登记策略，支持运行时热更换（swap），不改存储层。

    默认四个 op 都指向同一个 SimpleLifecyclePolicy 实例；
    你可以为任一 op 换成自己的策略（如“Read 即续命”“Update 必缩短”等）。
    """

    def __init__(
        self,
        default_policy: Optional[VectorLifecyclePolicy] = None,
    ) -> None:
        base = default_policy or SimpleLifecyclePolicy()
        self._policies: Dict[LifecycleOp, VectorLifecyclePolicy] = {
            op: base for op in LifecycleOp
        }
        self._lock = threading.RLock()

    def get(self, op: LifecycleOp) -> VectorLifecyclePolicy:
        with self._lock:
            return self._policies[op]

    def swap(self, op: LifecycleOp, policy: VectorLifecyclePolicy) -> None:
        """热更换某个 CRUD 操作的生命周期策略。"""
        with self._lock:
            self._policies[op] = policy

    def set_default(self, policy: VectorLifecyclePolicy) -> None:
        """把默认策略一次性替换（已有按 op 定制的策略保持不变）。"""
        with self._lock:
            for op in LifecycleOp:
                if self._policies[op].name == "simple-expiry":
                    self._policies[op] = policy

    def snapshot(self) -> Dict[str, str]:
        with self._lock:
            return {op.value: p.name for op, p in self._policies.items()}


class VectorLifecycleManager:
    """给一批上下文向量统一管理生命周期的门面。

    - 每个 event_id 一条 LifecycleMeta；
    - 所有 CRUD 操作都先经对应策略，策略本身可热更换；
    - 提供：create / read / update / delete / expires_at / is_expired /
      sweep(找过期) / touch(重置死期=续命) / set_duration(改持续时间)。
    """

    def __init__(
        self,
        registry: Optional[LifecyclePolicyRegistry] = None,
        default_duration_sec: float = 1800.0,
    ) -> None:
        self.registry = registry or LifecyclePolicyRegistry(
            SimpleLifecyclePolicy(default_duration_sec)
        )
        self._metas: Dict[str, LifecycleMeta] = {}
        self._lock = threading.RLock()

    # ---------- CRUD 生命周期操作（全部走策略） ----------

    def create(
        self,
        event_id: str,
        now: Optional[float] = None,
        params: Optional[Dict[str, Any]] = None,
    ) -> LifecycleMeta:
        now = time.time() if now is None else float(now)
        meta = LifecycleMeta(event_id=event_id, created_at=now, deadline_base=now)
        meta = self.registry.get(LifecycleOp.CREATE).on_create(meta, now, params)
        with self._lock:
            self._metas[event_id] = meta
        return meta

    def read(self, event_id: str, now: Optional[float] = None) -> Optional[LifecycleMeta]:
        now = time.time() if now is None else float(now)
        with self._lock:
            meta = self._metas.get(event_id)
            if meta is None:
                return None
            self.registry.get(LifecycleOp.READ).on_read(meta, now)
            return meta

    def update(
        self,
        event_id: str,
        params: Optional[LifecycleUpdateParams] = None,
        now: Optional[float] = None,
    ) -> Optional[LifecycleMeta]:
        now = time.time() if now is None else float(now)
        with self._lock:
            meta = self._metas.get(event_id)
            if meta is None:
                return None
            self.registry.get(LifecycleOp.UPDATE).on_update(meta, now, params)
            return meta

    def delete(self, event_id: str, now: Optional[float] = None) -> bool:
        now = time.time() if now is None else float(now)
        with self._lock:
            meta = self._metas.get(event_id)
            if meta is None:
                return False
            self.registry.get(LifecycleOp.DELETE).on_delete(meta, now)
            self._metas.pop(event_id, None)
            return True

    # ---------- 常用便利函数（对“向量生命周期”的直接管理） ----------

    def touch(self, event_id: str, now: Optional[float] = None) -> bool:
        """重置死期(deadline_base)=now，持续时间不变 —— “被用就续命”。"""
        return (
            self.update(
                event_id,
                LifecycleUpdateParams(reset_deadline=True),
                now=now,
            )
            is not None
        )

    def set_duration(
        self,
        event_id: str,
        duration_sec: float,
        now: Optional[float] = None,
    ) -> Optional[LifecycleMeta]:
        """显式修改“持续时间”（一般不发生）；保留原死期基准。"""
        return self.update(
            event_id,
            LifecycleUpdateParams(duration_sec=duration_sec),
            now=now,
        )

    def get(self, event_id: str) -> Optional[LifecycleMeta]:
        with self._lock:
            meta = self._metas.get(event_id)
            return meta

    def expires_at(self, event_id: str) -> Optional[float]:
        meta = self.get(event_id)
        return meta.expires_at() if meta else None

    def is_expired(self, event_id: str, now: Optional[float] = None) -> bool:
        now = time.time() if now is None else float(now)
        with self._lock:
            meta = self._metas.get(event_id)
            if meta is None:
                return True
            return self.registry.get(LifecycleOp.READ).is_expired(meta, now) \
                or meta.is_expired(now)

    def sweep(self, now: Optional[float] = None) -> List[str]:
        """返回此刻已过期(event 该删)的 event_id 列表（不在这里删数据）。"""
        now = time.time() if now is None else float(now)
        with self._lock:
            expired: List[str] = []
            for event_id, meta in self._metas.items():
                policy = self.registry.get(LifecycleOp.READ)
                if policy.is_expired(meta, now) or meta.is_expired(now):
                    expired.append(event_id)
            return expired

    def count(self) -> int:
        with self._lock:
            return len(self._metas)

    def meta_dicts(self) -> List[Dict[str, Any]]:
        with self._lock:
            return [m.to_dict() for m in self._metas.values()]

    def strategy_snapshot(self) -> Dict[str, str]:
        return self.registry.snapshot()
