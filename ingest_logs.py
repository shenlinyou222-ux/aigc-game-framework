"""把 Godot game_events.jsonl 批量导入日志向量库（批量嵌入版）。

用法:
    E:/py312/python.exe ingest_logs.py ^
        --source "%APPDATA%/Godot/app_userdata/StardewValley/logs/game_events.jsonl" ^
        --store log_vector_store.jsonl ^
        --embedder-model qwen3-embedding:0.6b

- DEBUG 级别不入库（日志级别决定是否写库）
- 嵌入按 16 条/批走 /api/embed 批量接口（实测 ~6 条/s vs 单条 0.42 条/s）
- 批量失败自动降级为逐条
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from aigc_core import LogVectorStore, OllamaLogEmbedder
from aigc_core.models import GameLogRecord

BATCH = 16


def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest Godot JSONL logs into LogVectorStore")
    parser.add_argument("--source", required=True, help="game_events.jsonl path")
    parser.add_argument("--store", default="log_vector_store.jsonl", help="vector store JSONL path")
    parser.add_argument("--limit", type=int, default=None, help="only read first N valid records")
    parser.add_argument(
        "--embedder-model",
        default="",
        help="Ollama embedding model, e.g. qwen3-embedding:0.6b or mxbai-embed-large; empty means no vector generation",
    )
    parser.add_argument("--embedder-url", default="http://localhost:11434", help="Ollama base URL")
    parser.add_argument(
        "--config",
        default="log_vector_lifecycle.json",
        help="single 布局的库生命周期配置",
    )
    parser.add_argument(
        "--layout",
        default="split",
        choices=["single", "split"],
        help="split=按读写特性三库(game_stores/*.db, 默认); single=单文件 --store",
    )
    parser.add_argument("--store-dir", default="game_stores", help="split 布局的库目录")
    args = parser.parse_args()

    embedder = (
        OllamaLogEmbedder(model=args.embedder_model, base_url=args.embedder_url)
        if args.embedder_model
        else None
    )
    if args.layout == "split":
        from aigc_core.multi_log_store import MultiLogStore

        store = MultiLogStore(args.store_dir)  # milestone/activity/dialogue 三库
    else:
        store = LogVectorStore(path=Path(args.store), config_path=args.config)

    # 1) 读取 + 校验 + 级别过滤
    source = Path(args.source)
    if not source.exists():
        raise FileNotFoundError(f"log source not found: {source}")
    records = []
    skipped_debug = 0
    with source.open("r", encoding="utf-8") as f:
        for line in f:
            if args.limit is not None and len(records) >= args.limit:
                break
            line = line.strip()
            if not line:
                continue
            try:
                record = GameLogRecord.model_validate(json.loads(line))
            except Exception:
                continue  # 坏行不阻塞整体导入
            if record.level == "DEBUG":
                skipped_debug += 1  # DEBUG 只留原始文件，不入库
                continue
            records.append(record)

    # 2) 批量嵌入缺向量的记录
    need = [r for r in records if not r.semantic_vector]
    preexisting = len(records) - len(need)
    embedded = 0
    t0 = time.time()
    if embedder and need:
        for i in range(0, len(need), BATCH):
            chunk = need[i : i + BATCH]
            try:
                vecs = embedder.embed_texts([r.to_embedding_text() for r in chunk])
                for r, v in zip(chunk, vecs):
                    r.semantic_vector = v
            except Exception as exc:
                print(f"  batch@{i} failed ({exc}), fallback to per-record", flush=True)
                for r in chunk:
                    try:
                        r.semantic_vector = [float(x) for x in embedder(r)]
                    except Exception:
                        pass
            print(
                f"  embedded {min(i + BATCH, len(need))}/{len(need)} ({time.time() - t0:.0f}s)",
                flush=True,
            )

    # 3) 入库
    added = store.add_many(records)
    layout_info = ""
    if args.layout == "split":
        cs = store.capacity_status()["stores"]
        layout_info = " | ".join(f"{k}={v['count']}" for k, v in cs.items())
    print(
        f"ingested={added} total={store.count()} "
        f"(vectors: batch_embedded={embedded}, preexisting={preexisting}) "
        f"skipped_debug={skipped_debug} elapsed={time.time() - t0:.1f}s "
        f"store={args.store_dir if args.layout == 'split' else store.path} {layout_info}"
    )


if __name__ == "__main__":
    main()
