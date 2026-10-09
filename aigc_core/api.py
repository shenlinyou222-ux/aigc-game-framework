"""FastAPI 入口：把主控能力暴露成 HTTP 接口。

游戏端和 AI 上层都通过同一组接口通信：
- POST /v1/events        游戏/感知组件写入事件
- POST /v1/memory/query  查询记忆
- POST /v1/logs          写入 Godot 日志到日志向量库
- POST /v1/logs/query    日志向量/过滤查询
- POST /v1/game/query    AI 只读查询
- POST /v1/game/command  AI/主控写操作
"""

from __future__ import annotations

import os
import threading
from pathlib import Path
from typing import Any, List, Optional

from fastapi import FastAPI
from pydantic import BaseModel

from .controller import MainController
from .entity_index import EntityIndex
from .lifecycle import VectorLifecycleManager
from .log_vector_store import LogVectorStore
from .models import (
    ActorRole,
    GameCommand,
    GameEvent,
    GameLogRecord,
    GameQuery,
    LogVectorQuery,
    MemoryQuery,
    WriterRole,
)

# 默认上下文向量库落盘路径：与 seed_current_session.py 使用同一份 JSONL，
# 这样“短 session 靠 /ctx/recall 恢复”在进程重启/跨进程后依然可用。
DEFAULT_CONTEXT_STORE_PATH = Path(__file__).resolve().parent.parent / "dsh_context_store.jsonl"
DEFAULT_CONTEXT_DURATION_SEC = 7 * 24 * 3600  # 默认 7 天存活，与 seed 脚本一致
# 06 机制：WorkingSet manifest 落盘目录（persist=True 时启用）
DEFAULT_WORKING_SET_DIR = Path(__file__).resolve().parent.parent / ".ctx_working_sets"


class EventWriteRequest(BaseModel):
    event: GameEvent
    writer: WriterRole = WriterRole.SENSOR


class MemoryQueryRequest(BaseModel):
    query: MemoryQuery


class LogWriteRequest(BaseModel):
    record: GameLogRecord


class LogQueryRequest(BaseModel):
    query: LogVectorQuery


class ContextIngestRequest(BaseModel):
    record: GameLogRecord


class ContextRecallRequest(BaseModel):
    query: LogVectorQuery


class ContextTouchRequest(BaseModel):
    event_id: str


class ContextSetDurationRequest(BaseModel):
    event_id: str
    duration_sec: float


# ---------- 06 机制：query_normalize / fact / tool / working-set / assemble ----------


class QueryNormalizeRequest(BaseModel):
    text: str


class FactIngestRequest(BaseModel):
    text: str
    event_id: Optional[str] = None
    tags: Optional[List[str]] = None


class FactSearchRequest(BaseModel):
    text: str
    limit: int = 5


class ToolSearchRequest(BaseModel):
    text: str
    top_k: int = 5


class WorkingSetRequest(BaseModel):
    session: str = ""


class WorkingSetLoadRequest(BaseModel):
    session: str = ""
    ids: List[Any]


class WorkingSetDropRequest(BaseModel):
    session: str = ""
    ids: List[str]


class AssembleContextRequest(BaseModel):
    session: str = ""
    user_text: str
    scope: Optional[List[str]] = None
    auto_load: bool = True


class GameQueryRequest(BaseModel):
    query: GameQuery
    actor: ActorRole = ActorRole.SYSTEM


class GameCommandRequest(BaseModel):
    command: GameCommand
    actor: ActorRole = ActorRole.SYSTEM


def create_app(
    controller: Optional[MainController] = None,
    *,
    persist: bool = True,
    store_path: Optional[str | Path] = None,
) -> FastAPI:
    # FastAPI 工厂。传入 controller 则直接使用（测试/定制入口）。
    # 未传时默认 persist=True：落到磁盘 JSONL（dsh_context_store.jsonl，与 seed 同库）
    # 并启用 7 天生命周期的死期+持续时间管理，保证重启/跨进程还能召回旧会话。
    # persist=False 保留旧的纯内存行为，供单测做状态隔离。
    app = FastAPI(title="AIGC Game Main Controller", version="0.0.1")
    ctx_embedder = None
    if controller is None:
        if persist:
            path = Path(store_path).expanduser() if store_path else DEFAULT_CONTEXT_STORE_PATH
            lifecycle = VectorLifecycleManager(default_duration_sec=DEFAULT_CONTEXT_DURATION_SEC)
            backend = os.environ.get("DSH_LOG_STORE_BACKEND", "jsonl").strip().lower()
            if backend == "sqlite":
                # 单文件 SQLite 后端：标签精确门控 + numpy 语义扫描（接口面与 jsonl 版一致）
                from .sqlite_log_store import SqliteLogStore

                log_store: Any = SqliteLogStore(
                    path=path.with_suffix(".db"), lifecycle=lifecycle
                )
            else:
                log_store = LogVectorStore(path=path, lifecycle=lifecycle)
            # persist 模式才挂嵌入器：ingest/recall 自动补 dense 向量（ollama 不可用时
            # 自动降级词法通道）；persist=False（单测）保持离线确定性。
            if os.environ.get("DSH_CTX_EMBED", "1") != "0":
                from .embedders import embed_text_best_effort

                ctx_embedder = embed_text_best_effort
            controller = MainController(
                log_store=log_store,
                working_set_dir=DEFAULT_WORKING_SET_DIR,
                ctx_embedder=ctx_embedder,
                # 实体层：与上下文库同目录的独立 JSON 索引（路径/工具/文件名精确召回）
                entity_index=EntityIndex(path=path.with_name("ctx_entities.json")),
            )
        else:
            controller = MainController()

    ctrl = controller

    # 固定能力：tool_registry/ 中的工具 schema 幂等同步进上下文向量库。
    # event_id 带内容哈希：未变更不重复入库；event_type=tool/schema 可经
    # /ctx/recall 与 /ctx/tool/search 召回。注册表缺失不阻断服务启动。
    # 注意：1) sync 必须在 ctrl 赋值之后（曾因引用未定义的 ctrl 被 except 静默吞掉）；
    #       2) 仅 persist 模式执行——单测的内存库要保持离线确定性，不被 66 条工具记录污染；
    #       3) 后台线程执行——每条记录一次 ollama 嵌入，同步整体可达分钟级，
    #          不能堵在 create_app 里（否则 uvicorn 迟迟不绑端口，探活全超时）。
    #       store/lifecycle 均带锁，与 HTTP 并发 ingest 安全。
    if persist:
        def _sync_registry_bg() -> None:
            try:
                from .tool_spec_loader import sync_registry_into_controller

                _reg_dir = Path(__file__).resolve().parent.parent / "tool_registry"
                _ids = sync_registry_into_controller(ctrl, _reg_dir)
                print(f"[tool_registry] synced {len(_ids)} tool-schema record(s) from {_reg_dir}", flush=True)
            except Exception as _exc:  # noqa: BLE001
                print(f"[tool_registry] sync skipped: {_exc}", flush=True)

        threading.Thread(target=_sync_registry_bg, name="tool-registry-sync", daemon=True).start()

    @app.get("/health")
    def health() -> dict:
        return {"status": "ok"}

    @app.post("/v1/events")
    def write_event(req: EventWriteRequest) -> dict:
        entry = ctrl.record_event(req.event, writer=req.writer)
        return {"ok": True, "event_id": entry.event_id}

    @app.post("/v1/memory/query")
    def query_memory(req: MemoryQueryRequest) -> dict:
        entries = ctrl.query_memory(req.query)
        return {"ok": True, "items": [e.model_dump() for e in entries]}

    @app.post("/v1/logs")
    def write_log(req: LogWriteRequest) -> dict:
        record = ctrl.record_log(req.record)
        return {"ok": True, "event_id": record.event_id}

    @app.post("/v1/logs/query")
    def query_logs(req: LogQueryRequest) -> dict:
        results = ctrl.query_logs(req.query)
        return {
            "ok": True,
            "items": [
                {"score": item.score, "record": item.record.model_dump()}
                for item in results
            ],
        }

    # ---------- DSH 上下文后端（/ctx/*） ----------

    @app.post("/ctx/ingest")
    def ctx_ingest(req: ContextIngestRequest) -> dict:
        record = ctrl.context_ingest(req.record)
        return {"ok": True, "event_id": record.event_id}

    @app.post("/ctx/recall")
    def ctx_recall(req: ContextRecallRequest) -> dict:
        # 查询侧 best-effort 补 dense 向量：调用方（插件/路由层）通常不带 query_vector
        if ctx_embedder is not None and not req.query.query_vector and req.query.text:
            req.query.query_vector = ctx_embedder(req.query.text[:2000])
        results = ctrl.context_recall(req.query)
        return {
            "ok": True,
            "items": [r.model_dump() for r in results],
        }

    @app.post("/ctx/touch")
    def ctx_touch(req: ContextTouchRequest) -> dict:
        ok = ctrl.touch_log(req.event_id)
        return {"ok": ok, "event_id": req.event_id}

    @app.post("/ctx/cleanup")
    def ctx_cleanup() -> dict:
        return {"ok": True, **ctrl.cleanup_logs()}

    @app.get("/ctx/status")
    def ctx_status() -> dict:
        resp: dict = {"ok": True, "status": ctrl.log_capacity_status()}
        # capacity_status 里的 retention_sec 是无 manager 时的遗留字段，易误读；
        # 真实存活策略以 lifecycle 管理器（死期+持续时间+touch）为准，这里一并返回。
        resp["lifecycle"] = ctrl.context_lifecycle_status()
        return resp

    @app.get("/ctx/lifecycle/status")
    def ctx_lifecycle_status() -> dict:
        return {"ok": True, **ctrl.context_lifecycle_status()}

    @app.get("/ctx/lifecycle/meta/{event_id}")
    def ctx_lifecycle_meta(event_id: str) -> dict:
        return {"ok": True, **ctrl.context_lifecycle_meta(event_id)}

    @app.post("/ctx/lifecycle/duration")
    def ctx_lifecycle_set_duration(req: ContextSetDurationRequest) -> dict:
        return ctrl.context_set_duration(req.event_id, req.duration_sec)

    @app.post("/ctx/lifecycle/reset")
    def ctx_lifecycle_reset(req: ContextTouchRequest) -> dict:
        return ctrl.context_reset_deadline(req.event_id)

    # ---------- 06 机制：query_normalize / fact / tool / working-set / assemble ----------

    @app.post("/ctx/query_normalize")
    def ctx_query_normalize(req: QueryNormalizeRequest) -> dict:
        return {"ok": True, "clauses": ctrl.normalize_query(req.text)}

    @app.post("/ctx/fact/ingest")
    def ctx_fact_ingest(req: FactIngestRequest) -> dict:
        record = ctrl.context_ingest_fact(req.text, event_id=req.event_id, tags=req.tags)
        return {"ok": True, "record": record}

    @app.post("/ctx/fact/search")
    def ctx_fact_search(req: FactSearchRequest) -> dict:
        return {"ok": True, "items": ctrl.fact_search(req.text, limit=req.limit)}

    @app.post("/ctx/tool/search")
    def ctx_tool_search(req: ToolSearchRequest) -> dict:
        return {"ok": True, "items": ctrl.tool_search(req.text, top_k=req.top_k)}

    @app.post("/ctx/working-set")
    def ctx_working_set(req: WorkingSetRequest) -> dict:
        return {"ok": True, "manifest": ctrl.working_set(req.session)}

    @app.post("/ctx/working-set/load")
    def ctx_working_set_load(req: WorkingSetLoadRequest) -> dict:
        return {"ok": True, **ctrl.working_set_load(req.session, req.ids)}

    @app.post("/ctx/working-set/drop")
    def ctx_working_set_drop(req: WorkingSetDropRequest) -> dict:
        return {"ok": True, **ctrl.working_set_drop(req.session, req.ids)}

    @app.post("/ctx/working-set/expand")
    def ctx_working_set_expand(req: WorkingSetRequest) -> dict:
        return {"ok": True, "items": ctrl.working_set_expand(req.session)}

    @app.post("/ctx/assemble")
    def ctx_assemble(req: AssembleContextRequest) -> dict:
        return {
            "ok": True,
            **ctrl.assemble_context(
                session=req.session,
                user_text=req.user_text,
                scope=req.scope,
                auto_load=req.auto_load,
            ),
        }

    @app.post("/v1/game/query")
    def game_query(req: GameQueryRequest) -> dict:
        result = ctrl.game_query(req.query, actor=req.actor)
        return result.model_dump()

    @app.post("/v1/game/command")
    def game_command(req: GameCommandRequest) -> dict:
        result = ctrl.game_command(req.command, actor=req.actor)
        return result.model_dump()

    @app.get("/v1/context")
    def context(player_id: str, location: str = "") -> dict:
        return {"ok": True, "data": ctrl.build_ai_context(player_id, location)}

    return app
