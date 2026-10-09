# -*- coding: utf-8 -*-
"""游戏向量库容量观测: 三库计数/上限/警戒线 + db 体积。"""
import sys
from pathlib import Path

from aigc_core.multi_log_store import MultiLogStore


def main(store_dir: str = "game_stores") -> int:
    if not Path(store_dir).exists():
        print(f"no store dir: {store_dir}")
        return 1
    m = MultiLogStore(store_dir)
    print(f"=== game vector stores ({store_dir}) ===")
    warn = False
    for name, st in zip(("milestone", "activity", "dialogue"), m._stores()):
        cs = st.capacity_status()
        db = Path(cs["db_path"])
        size_mb = db.stat().st_size / 1024 / 1024 if db.exists() else 0.0
        flag = " ⚠ OVER-WARNING" if cs["over_warning"] else ""
        warn = warn or cs["over_warning"]
        print(
            f"  {name:10s} count={cs['count']:<7} cap={cs['max_entries']:<7} "
            f"retention={cs['retention_sec']}s vec={cs.get('vec_count', 0):<7} "
            f"db={size_mb:.1f}MB{flag}"
        )
    cs = m.capacity_status()
    print(f"  total={cs['total']}")
    m.close()
    return 2 if warn else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1] if len(sys.argv) > 1 else "game_stores"))
