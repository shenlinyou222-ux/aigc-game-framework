"""working_set —— Session 级 WorkingSet manifest（上下文状态 = 一组 id）。

只记 id / kind / chars 等元信息，不存大文本；真正的大文本只在“本轮展开”时出现。
见 ../aigc_docs/06-会话上下文检索与工具组装机制.md 第 3 节。

数据形状：
    {
      "session": "s1",
      "loaded": [
        {"id": "fact/a1b2", "kind": "fact",  "chars": 260, "addedAt": ..., "lastUsed": ...},
        {"id": "event/s12x", "kind": "event", "chars": 90,  "addedAt": ..., "lastUsed": ...}
      ],
      "tokens": ...,
      "maxCount": 24,
      "maxTokens": 16000,
      "revision": 7
    }

- 程序维护：load/drop 只改这份清单，先过预算，再拉正文。
- 每轮组装 = manifest 轻量投影（锚点）+ 选中 id 的展开 + 本轮尾部。
- revision 每次 load/drop 自增，供审计/去重。
"""

from __future__ import annotations

import json
import threading
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

# token 估算：默认按字符数近似（demo 精度足够；正式可换成 tokenizer 计数）
TOKEN_PER_CHAR = 1.0


@dataclass
class WorkingItem:
    """清单里的一项：只存元信息，不存大文本。"""

    id: str
    kind: str = "event"            # fact / event / tool
    chars: int = 0
    added_at: float = field(default_factory=time.time)
    last_used: float = field(default_factory=time.time)

    def touch(self, now: Optional[float] = None) -> None:
        self.last_used = time.time() if now is None else float(now)


class BudgetError(Exception):
    """加入候选会超出条数/token 预算时报错（程序校验，先过预算再拉正文）。"""

    def __init__(self, reason: str, current: int, max_value: int) -> None:
        super().__init__(f"{reason}: current={current}, max={max_value}")
        self.reason = reason
        self.current = current
        self.max_value = max_value


class WorkingSetManifest:
    """纯数据 + 预算校验 + JSON 持久化；线程安全。"""

    def __init__(
        self,
        session: str = "",
        max_count: int = 24,
        max_tokens: int = 16000,
    ) -> None:
        self.session = session
        self.max_count = int(max_count)
        self.max_tokens = int(max_tokens)
        self.revision = 0
        self._items: Dict[str, WorkingItem] = {}
        self._lock = threading.RLock()

    # ---------- 只读 ----------

    @property
    def loaded(self) -> List[WorkingItem]:
        with self._lock:
            return [asdict(i) for i in self._items.values()]

    def count(self) -> int:
        with self._lock:
            return len(self._items)

    def tokens(self) -> int:
        with self._lock:
            return int(sum(i.chars * TOKEN_PER_CHAR for i in self._items.values()))

    def has(self, item_id: str) -> bool:
        with self._lock:
            return item_id in self._items

    def get(self, item_id: str) -> Optional[WorkingItem]:
        with self._lock:
            item = self._items.get(item_id)
            return WorkingItem(**asdict(item)) if item else None

    # ---------- 写 ----------

    def add_item(self, item_id: str, kind: str = "event", chars: int = 0) -> None:
        """加入/刷新清单（已存在则仅 touch）。先由调用方做预算校验。"""
        with self._lock:
            if item_id in self._items:
                self._items[item_id].touch()
                self._bump_locked()
                return
            self._items[item_id] = WorkingItem(
                id=item_id, kind=kind, chars=max(0, int(chars))
            )
            self._bump_locked()

    def drop(self, item_id: str) -> bool:
        with self._lock:
            existed = self._items.pop(item_id, None) is not None
            if existed:
                self._bump_locked()
            return existed

    def touch(self, item_id: str, now: Optional[float] = None) -> bool:
        with self._lock:
            item = self._items.get(item_id)
            if item is None:
                return False
            item.touch(now)
            return True

    # ---------- 预算校验 ----------

    def check_add(self, item_id: str, chars: int = 0) -> None:
        """加入前校验：不超条数上限、不超 token 上限。已存在视为 refresh（不算新增）。"""
        with self._lock:
            if item_id in self._items:
                return
            if len(self._items) >= self.max_count:
                raise BudgetError("working-set would exceed maxCount", len(self._items), self.max_count)
            if self.tokens() + int(chars) > self.max_tokens:
                raise BudgetError("working-set would exceed maxTokens", self.tokens(), self.max_tokens)

    # ---------- 投影 / 序列化 ----------

    def snapshot(self) -> Dict[str, Any]:
        """轻量投影（锚点）：只给 planner 看 id/kind/chars。"""
        with self._lock:
            return {
                "session": self.session,
                "loaded": [
                    {"id": i.id, "kind": i.kind, "chars": i.chars, "lastUsed": i.last_used}
                    for i in self._items.values()
                ],
                "count": len(self._items),
                "tokens": self.tokens(),
                "maxCount": self.max_count,
                "maxTokens": self.max_tokens,
                "revision": self.revision,
            }

    def to_dict(self) -> Dict[str, Any]:
        with self._lock:
            return {
                "session": self.session,
                "loaded": [asdict(i) for i in self._items.values()],
                "tokens": self.tokens(),
                "maxCount": self.max_count,
                "maxTokens": self.max_tokens,
                "revision": self.revision,
            }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "WorkingSetManifest":
        m = cls(
            session=data.get("session", ""),
            max_count=data.get("maxCount", 24),
            max_tokens=data.get("maxTokens", 16000),
        )
        with m._lock:
            for row in data.get("loaded", []):
                m._items[row["id"]] = WorkingItem(
                    id=row["id"],
                    kind=row.get("kind", "event"),
                    chars=int(row.get("chars", 0)),
                    added_at=float(row.get("added_at", row.get("addedAt", time.time()))),
                    last_used=float(row.get("last_used", row.get("lastUsed", time.time()))),
                )
            m.revision = int(data.get("revision", 0))
        return m

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> Optional["WorkingSetManifest"]:
        p = Path(path).expanduser()
        if not p.exists():
            return None
        try:
            return cls.from_dict(json.loads(p.read_text(encoding="utf-8")))
        except Exception:
            return None

    # ---------- internal ----------

    def _bump_locked(self) -> None:
        self.revision += 1


Resolver = Callable[[str, str], Optional[Dict[str, Any]]]
# resolver(kind, id) -> {"id", "kind", "text", "chars", "meta"}


class WorkingSetService:
    """把 manifest 与“真实正文存储”（fact/event 的 LogVectorStore、tool 的 InterfaceVectorStore）
    连起来：load 前取 chars 过预算，展开时把选中 id 拉成正文。

    不负责检索质量；它只是“组装 + 执行”的机械层。
    """

    def __init__(
        self,
        session: str = "",
        resolver: Optional[Resolver] = None,
        manifest: Optional[WorkingSetManifest] = None,
        path: Optional[Path] = None,
    ) -> None:
        self.session = session
        self.resolver = resolver
        self._path = Path(path).expanduser() if path else None
        self.manifest = manifest or WorkingSetManifest(session=session)
        if self._path and self._path.exists():
            loaded = WorkingSetManifest.load(self._path)
            if loaded is not None:
                self.manifest = loaded

    def load(self, ids: List[Dict[str, str]]) -> Dict[str, Any]:
        """ids: [{"id": "fact/a1", "kind": "fact"}, ...]

        程序执行：过预算校验 -> 更新 manifest -> 拉正文（展开）。返回汇总。
        """
        result = {"loaded": [], "rejected": [], "expanded": []}
        for entry in ids:
            item_id = entry["id"]
            kind = entry.get("kind", "event")
            resolved = self._resolve(kind, item_id)
            if resolved is None:
                result["rejected"].append({"id": item_id, "kind": kind, "reason": "not_found"})
                continue
            chars = int(resolved.get("chars", 0))
            try:
                self.manifest.check_add(item_id, chars)
            except BudgetError as exc:
                result["rejected"].append({"id": item_id, "kind": kind, "reason": str(exc)})
                continue
            self.manifest.add_item(item_id, kind=kind, chars=chars)
            result["loaded"].append({"id": item_id, "kind": kind, "chars": chars})
        result["expanded"] = self.expand()
        self._persist()
        return result

    def drop(self, ids: List[str]) -> Dict[str, Any]:
        result = {"dropped": [], "missing": []}
        for item_id in ids:
            if self.manifest.drop(item_id):
                result["dropped"].append(item_id)
            else:
                result["missing"].append(item_id)
        self._persist()
        return result

    def touch(self, item_id: str) -> bool:
        ok = self.manifest.touch(item_id)
        if ok:
            self._persist()
        return ok

    def expand(self) -> List[Dict[str, Any]]:
        """展开选中 id -> 正文（manifest 轻量投影 + 选中 id 的展开）。

        返回 [{id, kind, text, chars, meta}]，按加入顺序稳定输出。
        """
        out: List[Dict[str, Any]] = []
        for row in self.manifest.loaded:
            resolved = self._resolve(row["kind"], row["id"])
            if resolved is None:
                continue
            out.append(
                {
                    "id": row["id"],
                    "kind": row["kind"],
                    "chars": row["chars"],
                    "text": resolved.get("text", ""),
                    "meta": resolved.get("meta", {}),
                }
            )
        return out

    def snapshot(self) -> Dict[str, Any]:
        return self.manifest.snapshot()

    def _resolve(self, kind: str, item_id: str) -> Optional[Dict[str, Any]]:
        if self.resolver is None:
            return None
        return self.resolver(kind, item_id)

    def _persist(self) -> None:
        if self._path:
            self.manifest.save(self._path)
