"""把“当前这个会话做了什么”写进向量上下文库（试试咸淡）。

⚠️ 分层提醒：本脚本写的是 **JSONL 离线 demo 库**（`LogVectorStore`），
不是真正的向量数据库。真正的持久化/到期清理应交给 `aigc_core/vector_db.py`
（如 ChromaDB）——装好真库后请改走 VectorDb 实现，而不是继续追加 JSONL。

用途（demo）：
- 把本会话沉淀的结论/产物位置以“上下文向量”形式持久化到
  `dsh_context_store.jsonl`（全局长期记忆，session_id 留空），
- 下个会话用同一份库加载并按需 recall，即可快速恢复上下文，不必重放整段对话。

运行：
    E:\\py312\\python.exe seed_current_session.py

说明：
- 不依赖 Ollama：用确定性 HashEmbedder 生成向量（跨进程可复现），
- 生命周期走“死期 deadline_base + 持续时间 duration”（默认 7 天），可热更换策略。
"""

from __future__ import annotations

import argparse
from pathlib import Path

from aigc_core.embedders import OllamaLogEmbedder
from aigc_core.interface_store import HashEmbedder
from aigc_core.lifecycle import VectorLifecycleManager
from aigc_core.log_vector_store import LogVectorStore
from aigc_core.models import GameLogRecord

DEFAULT_STORE = "dsh_context_store.jsonl"
SESSION_ID = ""          # 空 = 全局长期记忆，任意 session 都可召回
DEFAULT_DURATION_SEC = 7 * 24 * 3600   # 7 天
DEFAULT_MODEL = "qwen3-embedding:0.6b" # 本机 Ollama 已装（见 `ollama list`）；不存在则退回 HashEmbedder


def make_embedder(model: str):
    """返回 `text -> vector` 的统一可调用：有 Ollama 用真 embedding，否则退回 HashEmbedder。

    重要：Ollama 只负责“生成向量”，不负责存储/生命周期；
    存哪、到期谁删，仍由 vector_db.py（真向量库）决定。
    """
    try:
        ollama = OllamaLogEmbedder(model=model, timeout=8.0)
        probe = ollama.embed_text("可达性探测")
        if probe:
            print(f"[seed] using Ollama embedder: {model} (dim={len(probe)})")

            def _embed(text: str):
                return ollama.embed_text(text)
            return _embed
    except Exception as exc:  # noqa: BLE001 - 任何失败都退回假向量
        print(f"[seed] Ollama unavailable ({exc}); fallback to HashEmbedder")
    hasher = HashEmbedder()
    return lambda text: hasher(text)


def record(
    event_id: str,
    event_type: str,
    kind: str,
    text: str,
    data: dict | None = None,
) -> GameLogRecord:
    import time

    return GameLogRecord(
        event_id=event_id,
        event_type=event_type,
        kind=kind,
        unix_time=time.time(),
        session_id=SESSION_ID,
        data={"text": text, **(data or {})},
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="把本会话结论写成上下文向量（可指定库与 embedding 模型）")
    parser.add_argument("--store", default=DEFAULT_STORE, help="目标 JSONL 库文件（分库用，默认 dsh_context_store.jsonl）")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Ollama embedding 模型，默认 qwen3-embedding:0.6b")
    args = parser.parse_args()

    store_path = Path(args.store).expanduser()
    if not store_path.is_absolute():
        store_path = Path(__file__).resolve().parent / store_path

    embedder = make_embedder(args.model)
    lifecycle = VectorLifecycleManager(default_duration_sec=DEFAULT_DURATION_SEC)
    store = LogVectorStore(path=store_path, lifecycle=lifecycle)

    # 幂等：不重复写同 event_id
    existing = {r.event_id for r in store.all()}
    notes = [
        record(
            "meta/task",
            "user/message",
            "user/message",
            "任务：把向量上下文机制落地成 DSH 插件 —— 要求每次输入/输出/调用都留痕，研究一套准确召回方法，并给上下文向量加死期+持续时间的生命周期管理。",
        ),
        record(
            "meta/conclusion-dsh",
            "assistant/message",
            "assistant/message",
            "结论：DSH = DeepSeek Harness（Cordis/TS）；插件订阅 ctx session/event 即可拿到 user/message、assistant/message、tool/call、tool/result，每条都有 seq 和毫秒 time。Python 后端 /ctx/* 负责向量化/召回/生命周期，插件只做薄桥。",
        ),
        record(
            "meta/conclusion-recall",
            "assistant/message",
            "assistant/message",
            "准确召回 = 混合 dense 向量 + 词法(BM25-lite) + 时效半衰 + 本地(session) + 角色/证据加权；实现于 aigc_demo_framework 的 LogVectorStore.recall_hybrid()，返回带 dense/lexical/recency/local/matched_terms 子分可调参审计。",
        ),
        record(
            "meta/phase-a-done",
            "tool/result",
            "tool/result",
            "阶段A已落地：aigc_demo_framework 加 session_id/kind、/ctx/ingest recall touch cleanup status、HybridRecallResult、tests/test_context.py；全量单测 105 通过。",
        ),
        record(
            "meta/lifecycle-done",
            "tool/result",
            "tool/result",
            "向量生命周期口子已落地：aigc_core/lifecycle.py 用 死期 deadline_base + 持续时间 duration，deadline_base+duration=理论删除时刻 expires_at；CRUD(Create/Read/Update/Delete)四操作各有策略(SimpleLifecyclePolicy)，LifecyclePolicyRegistry.swap 支持运行时热更换；LogVectorStore(lifecycle=...) 启用 + /ctx/lifecycle/status meta duration reset 端点；tests/test_lifecycle.py 全量 117 通过。",
        ),
        record(
            "meta/plugin-draft",
            "tool/result",
            "tool/result",
            "DSH 插件源码草稿：dsh-context-memory/（package.json + src/index.ts + tsconfig/tsdown + README），订阅 session/event → 翻译成 /ctx/ingest，并提供 /ctx/recall + /ctx/touch 组装函数；injectRecall 默认关；待放入 deepseek-harness-master/packages/context/ 后 pnpm install && pnpm build 类型检查。",
        ),
        record(
            "meta/next-session-hint",
            "user/message",
            "user/message",
            "给下个会话：请加载 --store 指定的库（全局长期记忆），按需 /ctx/recall 或直接读 04 文档；继续：把 dsh-context-memory 放进 harness 构建、phase C agent/pre-step 注入、查询改写/rerank、callId 成组、分层记忆。",
        ),
    ]

    added = 0
    for note in notes:
        if note.event_id in existing:
            continue
        # 生成语义向量：有 Ollama 用真 embedding，否则退回 HashEmbedder（确定性、跨进程可复现）
        note.semantic_vector = embedder(note.to_embedding_text())
        store.add(note)
        # 把“本次是给下个会话用的笔记”设成较长存活，避免过早被清
        store.set_lifecycle_duration(note.event_id, DEFAULT_DURATION_SEC)
        added += 1

    print(f"store: {store_path}")
    print(f"total records: {store.count()} (added this run: {added})")
    print("lifecycle enabled:", store.lifecycle is not None)
    if store.lifecycle is not None:
        print("lifecycle strategies:", store.lifecycle.strategy_snapshot())


if __name__ == "__main__":
    main()
