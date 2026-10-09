"""interface_store.py —— “接口即向量”的特殊向量库。

把 AI 可调用的“内部接口/命令”（schema + 描述）向量化，存进一个专门向量库，
让 AI 用自然语言语义召回“该调哪个接口”，再真正去执行。
与日志事件向量库不同：这里装的是“能力/接口定义”，不是过程日志。
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from pydantic import BaseModel, Field

Embedder = Callable[[str], List[float]]


class InterfaceSpec(BaseModel):
    """一个可调用接口的定义（也用于向量化召回）。"""

    name: str = Field(..., description="接口名/命令名，如 play_music")
    description: str = Field(default="", description="人类/LLM 可读的功能描述")
    category: str = Field(default="command", description="command / query / log / local_node")
    input_schema: Dict[str, Any] = Field(default_factory=dict, description="入参 JSON Schema")
    output_schema: Dict[str, Any] = Field(default_factory=dict, description="出参 JSON Schema")
    # 端到端执行的可选项：marker 字段由具体注册方补充，不参与向量
    meta: Dict[str, Any] = Field(default_factory=dict)

    def to_embedding_text(self) -> str:
        return json.dumps(
            {
                "name": self.name,
                "category": self.category,
                "description": self.description,
                "input": self.input_schema,
                "output": self.output_schema,
            },
            ensure_ascii=False,
            sort_keys=True,
        )


@dataclass
class InterfaceHit:
    spec: InterfaceSpec
    score: float


class HashEmbedder:
    """确定性哈希 embedding（无外部服务时用于 demo/测试）。

    用法与 OllamaLogEmbedder 一致：call(text) -> List[float]。
    """

    def __init__(self, dim: int = 96, seed: str = "mcel-interface") -> None:
        self.dim = dim
        self.seed = seed

    def _index(self, token: str) -> int:
        h = hashlib.md5((self.seed + token).encode("utf-8")).digest()
        return int.from_bytes(h[:4], "little") % self.dim

    def _sign(self, token: str) -> float:
        return 1.0 if int.from_bytes(hashlib.md5(("s" + token).encode()).digest()[:2], "big") % 2 == 0 else -1.0

    def __call__(self, text: str) -> List[float]:
        vec = [0.0] * self.dim
        for tok in self._tokens(text):
            vec[self._index(tok)] += self._sign(tok)
        norm = math.sqrt(sum(v * v for v in vec))
        if norm:
            vec = [v / norm for v in vec]
        return vec

    @staticmethod
    def _tokens(text: str) -> List[str]:
        """分词：ASCII 单词 + 中文按字/双字组合（支持中文语义召回）。"""
        out = list(re.findall(r"[a-zA-Z0-9_]+", text.lower()))
        cjk = re.findall(r"[\u4e00-\u9fff]+", text)
        for phrase in cjk:
            if len(phrase) == 1:
                out.append(phrase)
                continue
            # 单字 + 相邻双字组合
            out.append(phrase)
            for i in range(len(phrase)):
                out.append(phrase[i])
                if i + 1 < len(phrase):
                    out.append(phrase[i : i + 2])
        return out


class InterfaceVectorStore:
    """线程安全、轻量的“接口向量库”，JSONL 持久化。"""

    def __init__(
        self,
        path: Optional[os.PathLike[str] | str] = None,
        embedder: Optional[Embedder] = None,
    ) -> None:
        self._path = Path(path).expanduser() if path else None
        self._embedder = embedder or HashEmbedder()
        self._entries: Dict[str, Dict[str, Any]] = {}  # name -> {spec, vector}
        self._lock = threading.RLock()
        if self._path and self._path.exists():
            self._load(self._path)

    def add(self, spec: InterfaceSpec, embedder: Optional[Embedder] = None) -> InterfaceSpec:
        vector = (embedder or self._embedder)(spec.to_embedding_text())
        with self._lock:
            self._entries[spec.name] = {
                "spec": spec.model_copy(deep=True),
                "vector": [float(x) for x in vector],
            }
            if self._path:
                self._path.parent.mkdir(parents=True, exist_ok=True)
                self._save_locked(self._path)
            return spec

    def add_many(self, specs: List[InterfaceSpec]) -> int:
        for s in specs:
            self.add(s)
        return len(specs)

    def get(self, name: str) -> Optional[InterfaceSpec]:
        with self._lock:
            e = self._entries.get(name)
            return e["spec"].model_copy(deep=True) if e else None

    def names(self) -> List[str]:
        with self._lock:
            return sorted(self._entries.keys())

    def count(self) -> int:
        with self._lock:
            return len(self._entries)

    def query(self, text: str, embedder: Optional[Embedder] = None, top_k: int = 5) -> List[InterfaceHit]:
        qv = (embedder or self._embedder)(text)
        hits: List[InterfaceHit] = []
        with self._lock:
            for name, e in self._entries.items():
                score = _cosine(qv, e["vector"])
                if score is not None:
                    hits.append(InterfaceHit(spec=e["spec"].model_copy(deep=True), score=round(score, 6)))
        hits.sort(key=lambda h: h.score, reverse=True)
        return hits[:top_k]

    def save(self, path: Optional[os.PathLike[str] | str] = None) -> None:
        target = Path(path).expanduser() if path else self._path
        if target is None:
            raise ValueError("save() requires a path")
        with self._lock:
            self._save_locked(target)
            self._path = target

    # ---------- internal ----------

    def _load(self, path: Path) -> None:
        with self._lock:
            with path.open("r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        data = json.loads(line)
                        spec = InterfaceSpec.model_validate(data["spec"])
                        self._entries[spec.name] = {
                            "spec": spec,
                            "vector": [float(x) for x in data.get("vector", [])],
                        }
                    except Exception:
                        continue

    def _save_locked(self, path: Path) -> None:
        tmp = path.with_suffix(path.suffix + ".tmp")
        with tmp.open("w", encoding="utf-8") as f:
            for name, e in self._entries.items():
                f.write(
                    json.dumps(
                        {"spec": e["spec"].model_dump(), "vector": e["vector"]},
                        ensure_ascii=False,
                    )
                    + "\n"
                )
        tmp.replace(path)


def _cosine(a: List[float], b: List[float]) -> Optional[float]:
    if len(a) != len(b) or not a:
        return None
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0.0 or nb == 0.0:
        return None
    return dot / (na * nb)


def default_interface_specs() -> List[InterfaceSpec]:
    """把当前真实可用的命令/接口定义登记成一份默认列表（供向量召回）。"""
    return [
        InterfaceSpec(
            name="play_music",
            category="command",
            description="切换游戏背景音乐到指定曲目（纯歌曲歌单 day/night/cozy/tense/custom 等），可调音量；环境音请用 play_ambient",
            input_schema={
                "type": "object",
                "properties": {
                    "track": {"type": "string",
                              "enum": ["day", "night", "night2", "cozy", "cozy2",
                                       "tense", "tense2", "day2", "rain2", "custom1", "custom2"],
                              "description": "曲目名(纯歌曲, 不含环境音)"},
                    "volume_db": {"type": "number", "description": "音量分贝，可选"},
                },
                "required": ["track"],
            },
            output_schema={"type": "object", "properties": {"track": {"type": "string"}}},
        ),
        InterfaceSpec(
            name="play_ambient",
            category="command",
            description="开关游戏背景环境音（如 rain 雨声循环），与背景音乐并行播放互不打断；name=off 关闭",
            input_schema={
                "type": "object",
                "properties": {
                    "name": {"type": "string", "enum": ["rain", "off"],
                             "description": "环境音名, off=关闭当前环境音"},
                    "volume_db": {"type": "number", "description": "音量分贝，可选"},
                },
                "required": ["name"],
            },
            output_schema={"type": "object", "properties": {"ambient": {"type": "string"}}},
        ),
        InterfaceSpec(
            name="create_item",
            category="command",
            description="运行时新增一种物品种类（贴图复用已有资源、可自定义名字），仅 system 可调",
            input_schema={
                "type": "object",
                "properties": {
                    "item_id": {"type": "string"},
                    "name": {"type": "string"},
                    "icon_path": {"type": "string", "description": "复用已有贴图资源路径"},
                    "count": {"type": "integer"},
                },
                "required": ["item_id", "name", "icon_path"],
            },
            output_schema={"type": "object", "properties": {"item_id": {"type": "string"}}},
        ),
        InterfaceSpec(
            name="give_item",
            category="command",
            description="把指定物品给某玩家（可给自定义物品 item_id 加数量）",
            input_schema={
                "type": "object",
                "properties": {
                    "player_id": {"type": "string"},
                    "item_id": {"type": "string"},
                    "count": {"type": "integer"},
                },
                "required": ["player_id", "item_id"],
            },
            output_schema={"type": "object", "properties": {"item": {"type": "object"}}},
        ),
        InterfaceSpec(
            name="send_message",
            category="command",
            description="给玩家发一条系统/助手消息",
            input_schema={
                "type": "object",
                "properties": {"player_id": {"type": "string"}, "text": {"type": "string"}},
                "required": ["player_id", "text"],
            },
            output_schema={"type": "object", "properties": {"player_id": {"type": "string"}}},
        ),
        InterfaceSpec(
            name="set_game_setting",
            category="command",
            description="改游戏设置（如音量 volume / 语言 language）",
            input_schema={
                "type": "object",
                "properties": {"key": {"type": "string"}, "value": {"type": "any"}},
                "required": ["key", "value"],
            },
            output_schema={"type": "object", "properties": {"key": {"type": "string"}}},
        ),
        InterfaceSpec(
            name="query_logs",
            category="query",
            description="按向量/事件类型/时间查询游戏日志向量库",
            input_schema={
                "type": "object",
                "properties": {
                    "query_vector": {"type": "array", "items": {"type": "number"}},
                    "event_type": {"type": "string"},
                    "limit": {"type": "integer"},
                },
            },
            output_schema={"type": "object"},
        ),
    ]
