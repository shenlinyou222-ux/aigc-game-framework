# -*- coding: utf-8 -*-
"""可续跑入库：跳过已有向量的记录，只补缺失部分。"""
import json, time
from pathlib import Path
from aigc_core.log_vector_store import LogVectorStore
from aigc_core import OllamaLogEmbedder
from aigc_core.models import GameLogRecord

src = Path(r'C:\Users\user\AppData\Roaming\Godot\app_userdata\StardewValley\logs\game_events.jsonl')
store = LogVectorStore(path='log_vector_store.jsonl')
embedder = OllamaLogEmbedder(model='qwen3-embedding:0.6b', base_url='http://localhost:11434')

done = skipped = 0
t0 = time.time()
with src.open(encoding='utf-8') as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        try:
            record = GameLogRecord.model_validate(json.loads(line))
        except Exception:
            continue
        if store.get(record.event_id) is not None:
            skipped += 1
            continue
        try:
            record.semantic_vector = [float(x) for x in embedder(record)]
        except Exception as e:
            print('embed fail', record.event_id, e, flush=True)
        store.add(record)
        done += 1
        if done % 10 == 0:
            print(f'done={done} skipped={skipped} elapsed={time.time()-t0:.0f}s', flush=True)
print(f'FINAL: done={done} skipped={skipped} total_store={store.count()} elapsed={time.time()-t0:.0f}s', flush=True)
