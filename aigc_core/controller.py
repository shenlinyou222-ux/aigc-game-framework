"""主控（调度核心）。

主控不做 AI 推理，只负责：
1. 接收游戏事件并写入公共向量知识库（按写入者/生命周期做权限校验）；
2. 对外提供记忆查询；
3. 作为 AI 上层与真实游戏之间的唯一操作通道，做命令权限过滤；
4. 保留事件订阅钩子，方便后续把事件异步送入 AI/生成任务队列；
5. 接收/查询原始游戏日志向量库（Godot game_events.jsonl 的 Python 侧入口）。
"""

from __future__ import annotations

import re
import threading
from typing import Any, Callable, Dict, List, Optional

from .game import GameAdapter, MockGame
from .lifecycle import VectorLifecycleManager
from .log_vector_store import LogVectorStore
import time

from .models import GameLogRecord
from .memory import InMemoryMemoryStore, MemoryStoreError
from pathlib import Path

from . import planner as planner_mod
from .entity_index import EntityIndex
from .fact_store import FactStore
from .interface_store import InterfaceVectorStore, default_interface_specs
from .query_normalize import QueryNormalizer
from .working_set import WorkingSetService
from .models import (
    ActionResult,
    ActorRole,
    GameCommand,
    GameEvent,
    GameLogRecord,
    GameQuery,
    HybridRecallResult,
    LogSearchResult,
    LogVectorQuery,
    MemoryEntry,
    MemoryQuery,
    SourceType,
    WriterRole,
)


class MainController:
    """面向游戏和 AI 的主控入口。"""

    # 小灰助手（ASSISTANT）允许的只读查询
    ASSISTANT_QUERIES = {
        "get_world_state",
        "get_player_state",
        "get_inventory",
        "get_npc_state",
        "get_game_setting",
        "list_entities",
    }

    # 小灰助手允许的有限操作
    ASSISTANT_COMMANDS = {
        "send_message",
        "send_npc_dialogue",
        "show_emote",
        "organize_inventory",
        "set_game_setting",
        "play_music",
        # 存档无破坏性(只是快照), assistant 可触发; 读档仍仅 system
        "save_game",
    }

    # 主控/系统允许的操作（游戏本身不受这里限制，但真实接入时由适配器再判断）
    SYSTEM_COMMANDS = ASSISTANT_COMMANDS | {
        "give_item",
        "set_npc_confidence",
        "advance_mainline",
        # 创建“新物品种类”属于程序/数据修改，只允许 system 落
        "create_item",
        # 存档(无破坏性快照, 07 结论1)/读档(改写整个世界)
        "save_game",
        "load_game",
    }

    def __init__(
        self,
        memory_store: Optional[InMemoryMemoryStore] = None,
        game_adapter: Optional[GameAdapter] = None,
        log_store: Optional[LogVectorStore] = None,
        log_lifecycle: Optional[VectorLifecycleManager] = None,
        tool_store: Optional[InterfaceVectorStore] = None,
        fact_store: Optional[FactStore] = None,
        normalizer: Optional[QueryNormalizer] = None,
        working_set_dir: Optional[str | Path] = None,
        ctx_embedder: Optional[Callable[[str], List[float]]] = None,
        entity_index: Optional[EntityIndex] = None,
    ) -> None:
        self.memory = memory_store or InMemoryMemoryStore()
        self.game = game_adapter or MockGame()
        if log_store is None:
            # 若提供了生命周期管理器，则启用“死期+持续时间”语义（否则保持默认行为）
            log_store = LogVectorStore(lifecycle=log_lifecycle)
        self.logs = log_store
        # ---- 06 机制：工具层 / 事实层 / 轻量清洗 / WorkingSet ----
        if tool_store is None:
            tool_store = InterfaceVectorStore()
            if tool_store.count() == 0:
                tool_store.add_many(default_interface_specs())
        self.tool_store = tool_store
        self.fact_store = fact_store or FactStore(self.logs)
        self.normalizer = normalizer or QueryNormalizer()
        self._working_set_dir = (
            Path(working_set_dir).expanduser() if working_set_dir else None
        )
        self._working_sets: Dict[str, WorkingSetService] = {}
        self._event_handlers: Dict[str, List[Callable[[GameEvent], None]]] = {}
        # /ctx 通路嵌入器（api.py persist 模式注入；None=纯词法，单测保持离线）
        self._ctx_embedder = ctx_embedder
        # 实体层（路径/工具/文件名精确召回的倒排索引；None=不启用实体加成）
        self._entity_index = entity_index
        self._lock = threading.RLock()

    # ---------- 事件写入 ----------

    def record_event(
        self,
        event: GameEvent,
        writer: WriterRole = WriterRole.SENSOR,
    ) -> MemoryEntry:
        """游戏/感知组件调用；主控负责落库前的生命周期校验。"""
        entry = event.to_memory_entry()
        try:
            self.memory.add(entry, writer=writer)
        except MemoryStoreError as exc:
            raise MemoryStoreError(f"record_event rejected: {exc}") from exc

        with self._lock:
            handlers = list(self._event_handlers.get(event.event_type, []))
        for handler in handlers:
            handler(event)
        return entry

    def on_event(self, event_type: str, handler: Callable[[GameEvent], None]) -> None:
        """订阅某类事件。AI/生成任务可在这里挂异步旁路。"""
        with self._lock:
            self._event_handlers.setdefault(event_type, []).append(handler)

    # ---------- 记忆查询/删除 ----------

    def query_memory(self, query: MemoryQuery) -> List[MemoryEntry]:
        """所有经过主控的查询；AI 层可调用。"""
        return self.memory.query(query)

    def get_memory(self, event_id: str) -> Optional[MemoryEntry]:
        return self.memory.get(event_id)

    def delete_memory(self, event_id: str, actor: ActorRole = ActorRole.SYSTEM) -> bool:
        """只有主控/系统可删，AI/助手不可删。"""
        if actor not in (ActorRole.SYSTEM, ActorRole.GAME):
            raise PermissionError("only system/game actor can delete memory")
        return self.memory.delete(event_id)

    # ---------- 日志向量库 ----------

    def record_log(self, record: GameLogRecord) -> GameLogRecord:
        """Godot 日志桥接入口：把一条 game_events.jsonl 记录写入日志向量库。"""
        return self.logs.add(record)

    def query_logs(self, query: LogVectorQuery) -> List[LogSearchResult]:
        """查询日志向量库；向量语义召回或按类型/时间过滤。"""
        return self.logs.query(query)

    def get_log(self, event_id: str) -> Optional[GameLogRecord]:
        return self.logs.get(event_id)

    def delete_log(self, event_id: str) -> bool:
        """删除日志仅用于调试/主控清理；AI 助手不直接调用。"""
        return self.logs.delete(event_id)

    def touch_log(self, event_id: str, at: Optional[float] = None) -> bool:
        """把某条日志向量标记为“已注入上下文”，刷新它的空闲计时基准。

        调用方应在把该向量真正塞进上层上下文之前调用；不覆盖原始 unix_time。
        """
        return self.logs.touch(event_id, at=at)

    def cleanup_logs(self) -> Dict[str, object]:
        """执行生命周期清理：删空闲超时记录 + 按条数上限淘汰。

        可被外部定时任务周期调用，也可依赖 add/query 时的懒删除。
        """
        swept = self.logs.sweep_expired()
        evicted = self.logs.evict_over_cap()
        return {"swept_expired": swept, "evicted_over_cap": evicted}

    # ---------- DSH 上下文（/ctx/*）入口 ----------

    def context_ingest(
        self,
        record: GameLogRecord,
        semantic_vector: Optional[List[float]] = None,
    ) -> GameLogRecord:
        """把 DSH 的一条上下文事件（输入/输出/调用/结果）写入日志向量库。

        与 `record_log` 同一条落库通道；这里显式收口给 /ctx/ingest 用，
        保证每次输入输出和调用都进同一个向量库并被生命周期管理。
        """
        if semantic_vector:
            record.semantic_vector = [float(x) for x in semantic_vector]
        elif self._ctx_embedder is not None and not record.semantic_vector:
            # 调用方没带向量（registry 同步/插件 ingest 常态）：best-effort 补嵌，
            # 失败返回 [] 走词法通道，不阻塞落库。
            text = record.data.get("text") if isinstance(record.data, dict) else None
            if text:
                record.semantic_vector = self._ctx_embedder(str(text)[:2000])
        stored = self.logs.add(record)
        self._link_record_entities(stored)
        return stored

    def _link_record_entities(self, record: GameLogRecord) -> None:
        """把知识类事件的文本实体链接进实体层（路径/工具/文件名精确召回用）。

        跳过 tool/call（参数噪声大）与空文本；文本截断 2000（与嵌入器一致）。
        链接本身是确定性正则抽取 + 倒排登记，失败不阻塞落库。
        """
        if not self._entity_index:
            return
        if record.event_type == "tool/call":
            return
        text = record.data.get("text") if isinstance(record.data, dict) else None
        if text:
            try:
                self._entity_index.link(str(text)[:2000], record.event_id)
            except Exception:  # noqa: BLE001 实体层是旁路，绝不阻断主链路
                pass

    def context_recall(self, query: LogVectorQuery) -> List[HybridRecallResult]:
        """准确召回：混合 dense + 词法 + 时效 + 本地/角色加权。

        供 /ctx/recall 使用；返回带子分的 HybridRecallResult。
        默认开少量时效半衰与本地/角色加权，让“当前这轮”更能找回最近、同会话、
        工具结果这类高价值上下文。
        """
        return self.logs.recall_hybrid(
            query_text=query.text,
            query_vector=query.query_vector,
            session_id=query.session_id,
            event_type=query.event_type,
            since=query.since,
            until=query.until,
            limit=query.limit,
            min_score=query.min_score,
            recency_decay_sec=1800.0,  # 与默认空闲保留窗口一致
            local_boost=0.3,
            role_weights={
                "user/message": 0.1,
                "assistant/message": 0.0,
                "tool/call": 0.2,
                "tool/result": 0.3,
            },
            tags=query.tags,          # 标签精确门控：schema 已声明，须透传到存储层
            tag_match=query.tag_match,
            entity_index=self._entity_index,
            entity_boost_weight=0.5,  # 借鉴 mem0 的 ENTITY_BOOST_WEIGHT（实体层加成）
        )

    def log_capacity_status(self) -> Dict[str, object]:
        """返回日志向量库容量/警戒线/保留策略（供 /ctx/status）。"""
        return self.logs.capacity_status()

    # ---------- 向量生命周期（死期+持续时间，可热更换策略） ----------

    def context_lifecycle_status(self) -> Dict[str, object]:
        """上下文向量生命周期是否启用（需要 log_store 以 VectorLifecycleManager 构造）。"""
        if self.logs.lifecycle is None:
            return {"enabled": False, "reason": "lifecycle manager not attached"}
        return {
            "enabled": True,
            "count": self.logs.lifecycle.count(),
            "strategies": self.logs.lifecycle.strategy_snapshot(),
        }

    def context_lifecycle_meta(self, event_id: str) -> Dict[str, object]:
        """取某条上下文向量的生命周期元数据（含 expires_at）。"""
        meta = self.logs.lifecycle_meta(event_id)
        if meta is None:
            return {"event_id": event_id, "found": False}
        return {"event_id": event_id, "found": True, "meta": meta}

    def context_set_duration(self, event_id: str, duration_sec: float) -> Dict[str, object]:
        """显式修改某向量“持续时间”（保留死期基准），返回新的理论删除时刻。"""
        new_exp = self.logs.set_lifecycle_duration(event_id, duration_sec)
        if new_exp is None:
            return {"ok": False, "event_id": event_id, "reason": "lifecycle manager not attached"}
        return {"ok": True, "event_id": event_id, "expires_at": new_exp}

    def context_reset_deadline(self, event_id: str) -> Dict[str, object]:
        """重置某向量“死期基准”=now（续命，duration 不变）。"""
        ok = self.touch_log(event_id)
        return {"ok": ok, "event_id": event_id}


    # ---------- 06 机制：查询轻量清洗 / 事实层 / 工具层 / WorkingSet / 组装 ----------

    def normalize_query(self, text: str) -> List[str]:
        """轻量 NLP 预处理：返回 1~3 条 raw 子句，不改写语义（空/弱查询交给 fallback）。"""
        return self.normalizer.normalize(text)

    def context_ingest_fact(
        self,
        text: str,
        event_id: Optional[str] = None,
        tags: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """事实层入库：脏过程写压缩成干净长期事实（kind=fact，全局长期）。"""
        record = self.fact_store.ingest(text, event_id=event_id, tags=tags)
        if self._entity_index and text:
            self._entity_index.link(text[:2000], record.event_id)
        return record.model_dump()

    def fact_search(self, text: str, limit: int = 5) -> List[Dict[str, Any]]:
        """事实层召回（scope: fact）。"""
        hits = self.fact_store.search(text, limit=limit)
        return [
            {
                "id": h.record.event_id,
                "kind": "fact",
                "score": h.score,
                "snippet": (h.record.data.get("text") or "")[:200],
            }
            for h in hits
        ]

    def tool_search(self, text: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """工具层召回（scope: tool），附结构化 spec 供组装。

        改道（2026-09）：优先走 tool/schema 召回链——即 tool_registry/*.json 经
        sync_registry_* 入库到上下文向量库的 event_type="tool/schema" 记录
        （系统级工具：STT/ASR、blender、CLI 等都在这里）。命中为空时回退
        InterfaceVectorStore（游戏内工具），保证旧调用方不空手而归。
        """
        try:
            recalled = self.logs.recall_hybrid(
                query_text=text,
                event_type="tool/schema",
                limit=max(top_k, 5),
                recency_decay_sec=0.0,  # 工具 schema 不做时效衰减（90 天生命期已足够）
            )
        except Exception:
            recalled = []
        items: List[Dict[str, Any]] = []
        for hit in recalled[:top_k]:
            rec = getattr(hit, "record", None)
            data = dict(getattr(rec, "data", None) or {})
            tool = str(data.get("tool") or "")
            command = str(data.get("command") or "")
            tool_id = "toolspec/" + (tool + ":" + command if command else tool + ":overview")
            items.append({
                "id": tool_id,
                "kind": "tool/schema",
                "score": getattr(hit, "score", 0.0),
                "snippet": str(data.get("text") or "")[:200],
                "spec": {
                    "name": tool + ((" " + command) if command else ""),
                    "description": str(data.get("text") or "")[:400],
                    "category": data.get("category") or "",
                    "command": command,
                    "source_file": data.get("source_file") or "",
                    "input_schema": {},
                    "output_schema": {},
                    "meta": {"registry": True},
                },
            })
        if items:
            return items
        # 回退：游戏 InterfaceStore（旧链路）
        hits = self.tool_store.query(text, top_k=top_k)
        return [
            {
                "id": "tool/" + h.spec.name,
                "kind": "tool",
                "score": h.score,
                "snippet": (h.spec.description or "")[:200],
                "spec": h.spec.model_dump(),
            }
            for h in hits
        ]
    def working_set(self, session: str = "") -> Dict[str, Any]:
        """取某 session 的 WorkingSet manifest 轻量投影。"""
        return self._get_working_set(session).snapshot()

    def working_set_load(
        self,
        session: str,
        ids: List[Dict[str, str]],
    ) -> Dict[str, Any]:
        """程序执行 load：先过预算校验，再更新 manifest 并展开。"""
        entries: List[Dict[str, str]] = []
        for item in ids:
            if isinstance(item, dict):
                item_id = str(item["id"])
                kind = str(item.get("kind", ""))
            else:
                item_id = str(item)
                kind = ""
            if not kind:
                kind = planner_mod.split_id(item_id)["kind"]
            entries.append({"id": item_id, "kind": kind})
        return self._get_working_set(session).load(entries)

    def working_set_drop(self, session: str, ids: List[str]) -> Dict[str, Any]:
        """卸载：只改 manifest（不删底层正文）。"""
        return self._get_working_set(session).drop([str(i) for i in ids])

    def working_set_expand(self, session: str) -> List[Dict[str, Any]]:
        """展开选中 id -> 正文（manifest 投影 + 本轮组装用）。"""
        return self._get_working_set(session).expand()

    def recall_candidates(
        self,
        text: str,
        session: str = "",
        scope: Optional[List[str]] = None,
        limit: int = 8,
    ) -> List[Dict[str, Any]]:
        """统一召回：fact + event + tool 走同一套检索，按需一并查出。

        返回 [{id, kind, score, snippet, spec?}]，这就是“工具与事实同构、需要时一并查出”。
        """
        scope = list(scope) if scope else ["fact", "event", "tool"]
        clauses = self.normalizer.normalize(text)
        query_text = " ".join(clauses) or (text or "")
        out: List[Dict[str, Any]] = []

        if "fact" in scope:
            out.extend(self.fact_search(query_text, limit=limit))
        if "event" in scope:
            hits = self.logs.recall_hybrid(
                query_text=query_text,
                session_id=session or None,
                limit=limit,
                recency_decay_sec=1800.0,
                local_boost=0.3,
            )
            for h in hits:
                if (h.record.kind or "") == "fact":
                    continue  # 事实层已单独召回，不重复进 event 层
                out.append(
                    {
                        "id": h.record.event_id,
                        "kind": "event",
                        "score": h.score,
                        "snippet": (h.record.data.get("text") or "")[:200],
                    }
                )
        if "tool" in scope:
            out.extend(self.tool_search(query_text, top_k=limit))

        out.sort(key=lambda c: float(c.get("score", 0.0)), reverse=True)
        return out[:limit]

    def assemble_context(
        self,
        session: str = "",
        user_text: str = "",
        scope: Optional[List[str]] = None,
        auto_load: bool = True,
    ) -> Dict[str, Any]:
        """每轮预组装：normalize -> 统一召回 -> WorkingSet load -> 展开 + 工具 spec -> messages。

        这是“程序发起预组装 + 按需口子”的最小闭环，planner JSON 解析由 planner.parse_plan
        在接入真实 LLM 时替换 auto_load 的默认行为。
        """
        svc = self._get_working_set(session)
        clauses = self.normalizer.normalize(user_text)
        candidates = self.recall_candidates(user_text, session=session, scope=scope, limit=8)

        loaded_ids: List[str] = []
        if auto_load and candidates:
            # 默认把 top 命中（fact/event）载入工作集；工具只附 spec 不占 manifest 展开槽
            to_load = [c for c in candidates if c["kind"] in ("fact", "event")][:4]
            svc.load([{"id": c["id"], "kind": c["kind"]} for c in to_load])
            loaded_ids = [c["id"] for c in to_load]

        expansion = svc.expand()
        tool_specs = [c["spec"] for c in candidates if c.get("spec")]

        plan_input = planner_mod.build_planner_input(
            user_text,
            working_set_snapshot=svc.snapshot(),
            query_clauses=clauses,
        )
        messages = planner_mod.assemble_messages(
            expansion,
            user_text,
            tool_specs=tool_specs,
        )
        return {
            "session": session,
            "clauses": clauses,
            "candidates": candidates,
            "loaded_ids": loaded_ids,
            "working_set": svc.snapshot(),
            "planner_input": plan_input,
            "messages": messages,
        }

    # ---------- internal（06 机制） ----------

    def _get_working_set(self, session: str) -> WorkingSetService:
        with self._lock:
            svc = self._working_sets.get(session)
            if svc is None:
                path = None
                if self._working_set_dir is not None:
                    name = re.sub(r"[^0-9A-Za-z._-]", "_", session or "default") or "default"
                    path = self._working_set_dir / f"working-set-{name}.json"
                svc = WorkingSetService(
                    session=session,
                    resolver=self._resolve_working_item,
                    path=path,
                )
                self._working_sets[session] = svc
            return svc

    def _resolve_working_item(
        self,
        kind: str,
        item_id: str,
    ) -> Optional[Dict[str, Any]]:
        """WorkingSet 展开的解析器：kind=fact/event 查日志库，kind=tool 查工具库。"""
        if kind == "tool":
            name = item_id[len("tool/"):] if item_id.startswith("tool/") else item_id
            spec = self.tool_store.get(name)
            if spec is None:
                return None
            return {
                "id": item_id,
                "kind": kind,
                "text": spec.description or spec.name,
                "chars": len(spec.description or spec.name),
                "meta": {"name": spec.name, "category": spec.category},
            }
        record = self.logs.get(item_id)
        if record is None:
            return None
        text = record.data.get("text") or ""
        if not text:
            text = record.to_embedding_text()
        return {
            "id": item_id,
            "kind": kind,
            "text": text,
            "chars": len(text),
            "meta": {
                "event_type": record.event_type,
                "kind": record.kind,
                "unix_time": record.unix_time,
                "session_id": record.session_id,
            },
        }

    # ---------- 游戏操作接口 ----------

    def game_query(self, query: GameQuery, actor: ActorRole = ActorRole.SYSTEM) -> ActionResult:
        """AI 或上层调用的只读查询。"""
        if actor == ActorRole.ASSISTANT and query.query_type not in self.ASSISTANT_QUERIES:
            denied = ActionResult(
                ok=False,
                code="denied",
                message=f"assistant cannot query: {query.query_type}",
            )
            self.logs.add(
                GameLogRecord(
                    event_id=f"ai.query.{query.query_id}",
                    event_type="ai.query",
                    time=time.strftime("%Y-%m-%dT%H:%M:%S"),
                    unix_time=time.time(),
                    data={
                        "query": query.query_type,
                        "params": query.params,
                        "actor": actor.value,
                        "result": {"ok": False, "code": "denied", "message": denied.message},
                    },
                    tags=["src:ai", f"query:{query.query_type}", "err:denied"],
                )
            )
            return denied
        result = self.game.query(query)
        if not result.ok:
            # 只读查询默认不落库（build_ai_context 一次会打多个查询，量大会淹库），
            # 失败才记——同样的 debug 金矿。
            self.logs.add(
                GameLogRecord(
                    event_id=f"ai.query.{query.query_id}",
                    event_type="ai.query",
                    time=time.strftime("%Y-%m-%dT%H:%M:%S"),
                    unix_time=time.time(),
                    data={
                        "query": query.query_type,
                        "params": query.params,
                        "actor": actor.value,
                        "result": {"ok": result.ok, "code": result.code, "message": result.message},
                    },
                    tags=["src:ai", f"query:{query.query_type}", f"err:{result.code}"],
                )
            )
        return result

    def _log_ai_command(self, command: GameCommand, actor: ActorRole, result: ActionResult) -> None:
        """AI->游戏命令日志：无论成败都写入日志向量库（失败也是 debug 金矿 + 自我记忆）。"""
        self.logs.add(
            GameLogRecord(
                event_id=f"ai.command.{command.command_id}",
                event_type="ai.command",
                time=time.strftime("%Y-%m-%dT%H:%M:%S"),
                unix_time=time.time(),
                data={
                    "command": command.command_type,
                    "params": command.params,
                    "actor": actor.value,
                    "result": {"ok": result.ok, "code": result.code, "message": result.message},
                },
                tags=[
                    "src:ai",
                    f"cmd:{command.command_type}",
                    "ok" if result.ok else f"err:{result.code}",
                ],
            )
        )

    def game_command(self, command: GameCommand, actor: ActorRole = ActorRole.SYSTEM) -> ActionResult:
        """AI 或上层调用的写操作。真实游戏改动统一从这一入口走。"""
        if actor == ActorRole.ASSISTANT and command.command_type not in self.ASSISTANT_COMMANDS:
            denied = ActionResult(
                ok=False,
                code="denied",
                message=f"assistant cannot command: {command.command_type}",
            )
            self._log_ai_command(command, actor, denied)
            return denied
        if actor not in (ActorRole.SYSTEM, ActorRole.GAME, ActorRole.ASSISTANT):
            denied = ActionResult(ok=False, code="denied", message=f"unknown actor: {actor}")
            self._log_ai_command(command, actor, denied)
            return denied

        result = self.game.command(command)
        # AI->游戏调用日志：无论成败都写入日志向量库。
        # 失败命令（denied/error）是 debug 金矿，也是 AI 自我记忆的一部分
        # （"这个操作被拒过"，防止重复犯错）。tags 用受控词表便于精确检索：
        #   src:ai / cmd:<type> / ok | err:<code>
        self._log_ai_command(command, actor, result)
        # 写操作成功后，主控可补一条系统事件，方便上层感知“谁执行了什么”。
        if result.ok and command.command_type in self.SYSTEM_COMMANDS:
            self.record_event(
                GameEvent(
                    event_type="system.command_executed",
                    source_type=SourceType.SYSTEM,
                    source_faction="system",
                    entity_id=command.command_type,
                    payload={
                        "command": command.command_type,
                        "params": command.params,
                        "actor": actor.value,
                    },
                ),
                writer=WriterRole.PROGRAM,
            )
        return result

    # ---------- AI 接入点（上层实现） ----------

    def build_ai_context(
        self,
        player_id: str,
        location: str = "",
        limit: int = 10,
    ) -> Dict[str, Any]:
        """给上层 AI 组装上下文：世界状态 + 背包 + NPC + 相关记忆。"""
        world = self.game_query(
            GameQuery(query_type="get_world_state", params={"player_id": player_id})
        )
        player = self.game_query(
            GameQuery(query_type="get_player_state", params={"player_id": player_id})
        )
        inventory = self.game_query(
            GameQuery(query_type="get_inventory", params={"player_id": player_id})
        )
        memories = self.query_memory(
            MemoryQuery(location=location or None, limit=limit)
        )
        return {
            "world": world.data,
            "player": player.data,
            "inventory": inventory.data,
            "memories": [m.model_dump() for m in memories],
        }
