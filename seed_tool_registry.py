"""把 tool_registry/ 中的工具 schema 同步进 ctx 向量库（CLI）。

用法：
    python seed_tool_registry.py                       # HTTP 同步到 127.0.0.1:8000
    python seed_tool_registry.py --url http://127.0.0.1:8000
    python seed_tool_registry.py --duration-sec 7776000

幂等：内容不变 → 同 event_id 覆盖；内容变更 → 新 id（旧记录自然过期）。
与 seed_current_session.py 同一份 dsh_context_store 存储。
"""

from __future__ import annotations

import argparse
from pathlib import Path

from aigc_core.tool_spec_loader import (
    DEFAULT_DURATION_SEC,
    REGISTRY_EVENT_TYPE,
    sync_registry_http,
)

DEFAULT_REGISTRY_DIR = Path(__file__).resolve().parent / "tool_registry"
DEFAULT_BASE_URL = "http://127.0.0.1:8000"


def main() -> int:
    parser = argparse.ArgumentParser(description="Seed tool-schema registry into ctx vector store")
    parser.add_argument("--registry-dir", type=Path, default=DEFAULT_REGISTRY_DIR)
    parser.add_argument("--url", default=DEFAULT_BASE_URL)
    parser.add_argument("--duration-sec", type=float, default=DEFAULT_DURATION_SEC)
    args = parser.parse_args()

    ids = sync_registry_http(args.registry_dir, base_url=args.url, duration_sec=args.duration_sec)
    print(f"[seed_tool_registry] event_type={REGISTRY_EVENT_TYPE} synced={len(ids)} -> {args.url}")
    for event_id in ids:
        print(f"  - {event_id}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
