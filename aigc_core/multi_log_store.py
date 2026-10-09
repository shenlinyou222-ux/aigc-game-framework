# -*- coding: utf-8 -*-
"""MultiLogStore: 按读写特性分库的游戏日志门面。

三库划分(按读/写特性, 详见 README 讨论记录):
- milestone  MILESTONE 级(world.snapshot / game_started / 剧情节点) —— 只增不删,
             库容量天然极小(每局几条~几十条), 常驻参与所有查询, 是"世界基线"答案库
- activity   其余 INFO 玩法事件(world.*/player.tool_used/ai.command) —— 高频写,
             7 天空闲 TTL + LRU 上限滚动
- dialogue   npc.dialogue_line / story_dialog.* —— 中频交互触发, 长保留(30 天)

写入路由: level=MILESTONE → milestone; 对话类型 → dialogue; 其余 → activity。
读取: 默认三库扇出取候选 → 合并后**统一融合打分一次**(fuse_hybrid, 单库语义零漂移)。
标签零命中降级: 各库均空且带了 tags → 去标签重查, 结果标 tag_matched=False。
"""

from __future__ import annotations

import threading
import time
from pathlib import Path
from typing import Dict, List, Optional

from .log_vector_store import StoreLifecycleConfig, _cosine_similarity, fuse_hybrid
from .models import GameLogRecord, LogSearchResult, LogVectorQuery, HybridRecallResult
from .sqlite_log_store import SqliteLogStore

DIALOGUE_TYPES = {
    "npc.dialogue_line",
    "story_dialog.started",
    "story_dialog.finished",
    "story_dialog.cancelled",
    "story_dialog.interact_triggered",
}

# 三库生命周期(按读写特性独立配置)
MILESTONE_CONFIG = StoreLifecycleConfig(retention_sec=-1, max_entries=-1)   # 绝无删除
ACTIVITY_CONFIG = StoreLifecycleConfig(retention_sec=7 * 24 * 3600, max_entries=20000)
DIALOGUE_CONFIG = StoreLifecycleConfig(retention_sec=30 * 24 * 3600, max_entries=50000)


class MultiLogStore:
    """三库门面: 对外方法面与单库一致, 内部按级别/类型路由 + 扇出统一打分。"""

    def __init__(
        self,
        store_dir: str | Path = "game_stores",
        *,
        milestone=None,
        activity=None,
        dialogue=None,
    ) -> None:
        d = Path(store_dir)
        d.mkdir(parents=True, exist_ok=True)
        self.milestone = milestone or SqliteLogStore(path=d / "milestone.db", config=MILESTONE_CONFIG)
        self.activity = activity or SqliteLogStore(path=d / "activity.db", config=ACTIVITY_CONFIG)
        self.dialogue = dialogue or SqliteLogStore(path=d / "dialogue.db", config=DIALOGUE_CONFIG)
        self._lock = threading.RLock()

    # ---------- 路由 ----------
    def route(self, record: GameLogStore_record_hint):
        if record.level == "MILESTONE":
            return self.milestone
        if record.event_type in DIALOGUE_TYPES:
            return self.dialogue
        return self.activity

    def _stores(self):
        return [self.milestone, self.activity, self.dialogue]

    # ---------- 写入 ----------
    def add(self, record: GameLogRecord) -> GameLogRecord:
        return self.route(record).add(record)

    def add_many(self, records) -> int:
        n = 0
        for r in records:
            self.add(r)
            n += 1
        return n

    def delete(self, event_id: str) -> bool:
        for st in self._stores():
            if st.delete(event_id):
                return True
        return False

    # ---------- 读取 ----------
    def get(self, event_id: str):
        for st in self._stores():
            r = st.get(event_id)
            if r:
                return r
        return None

    def count(self) -> int:
        return sum(st.count() for st in self._stores())

    def all(self):
        out = []
        for st in self._stores():
            out.extend(st.all())
        return out

    def _gather(
        self,
        *,
        event_type=None, session_id=None, since=None, until=None,
        tags=None, tag_match="all",
    ):
        """门控统一语义: 任一库在门内命中 → 只收命中库的候选;
        带门但全库零命中 → 全局降级(去门重查), tag_matched=False。"""
        per = []
        for st in self._stores():
            recs, ok = st.fetch_meta_candidates(
                event_type=event_type, session_id=session_id,
                since=since, until=until, tags=tags, tag_match=tag_match,
            )
            per.append((recs, ok))
        if tags and any(ok for _, ok in per):
            merged = [r for recs, ok in per if ok for r in recs]
            return merged, True
        merged = [r for recs, _ in per for r in recs]
        return merged, (not bool(tags))

    def query(self, query: LogVectorQuery) -> List[LogSearchResult]:
        merged, tag_matched = self._gather(
            event_type=query.event_type, session_id=query.session_id,
            since=query.since, until=query.until,
            tags=query.tags, tag_match=query.tag_match,
        )
        if query.query_vector:
            results: List[LogSearchResult] = []
            for record in merged:
                if not record.semantic_vector:
                    continue
                score = _cosine_similarity(query.query_vector, record.semantic_vector)
                if score is None or score < query.min_score:
                    continue
                results.append(
                    LogSearchResult(record=record, score=round(score, 6), tag_matched=tag_matched)
                )
            results.sort(key=lambda x: (x.score, x.record.unix_time), reverse=True)
            return results[: query.limit]
        merged.sort(key=lambda r: r.unix_time, reverse=True)
        return [LogSearchResult(record=r, tag_matched=tag_matched) for r in merged[: query.limit]]

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
        prefilter_top: int = 5000,
    ) -> List[HybridRecallResult]:
        """三库扇出候选 → 合并 → 统一融合打分一次(跨库分数可比)。

        prefilter_top: 候选超限时 dense 预筛(词法精排只对预筛集),
        15 万条库的词法 15s 爆点由此兜底。"""
        now = time.time()
        merged, tag_matched = self._gather(
            event_type=event_type, session_id=session_id,
            since=since, until=until, tags=tags, tag_match=tag_match,
        )
        seen, uniq = set(), []
        for r in merged:
            if r.event_id not in seen:
                seen.add(r.event_id)
                uniq.append(r)
        return fuse_hybrid(
            uniq,
            now=now,
            query_text=query_text,
            query_vector=query_vector,
            session_id=session_id,
            limit=limit,
            min_score=min_score,
            dense_weight=dense_weight,
            lexical_weight=lexical_weight,
            recency_decay_sec=recency_decay_sec,
            recency_weight=recency_weight,
            local_boost=local_boost,
            role_weights=role_weights,
            tag_matched=tag_matched,
            prefilter_top=prefilter_top,
        )

    # ---------- 基线查询 ----------
    def world_baseline(self) -> Optional[GameLogRecord]:
        """最新世界快照(启动基线): '场上有什么'类问题的精确答案,
        从 milestone 库按 event_type 直取, 不走语义召回。"""
        hits = self.milestone.query(
            LogVectorQuery(event_type="world.snapshot", limit=1)
        )
        return hits[0].record if hits else None

    # ---------- 运维 ----------
    def capacity_status(self) -> Dict[str, object]:
        per = {}
        total = 0
        for name, st in zip(("milestone", "activity", "dialogue"), self._stores()):
            cs = st.capacity_status()
            per[name] = cs
            total += cs["count"]
        return {"backend": "multi-sqlite", "total": total, "stores": per}

    def sweep_expired(self, now: Optional[float] = None) -> int:
        return sum(st.sweep_expired(now=now) for st in self._stores())

    def close(self) -> None:
        for st in self._stores():
            st.close()


# 仅用于类型提示的别名(避免循环 import 的占位写法)
GameLogStore_record_hint = GameLogRecord
