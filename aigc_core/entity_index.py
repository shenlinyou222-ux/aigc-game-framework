# -*- coding: utf-8 -*-
"""entity_index —— 实体层：确定性实体抽取 + 实体↔记忆倒排 + 检索加成。

借鉴 mem0 的实体库思路（单独的 entity store、检索时给命中实体的记忆加权重），
但抽取规则重写成适配中文 / 路径 / 代码场景的确定性启发式，**不依赖 spaCy**：

- mem0 用 spaCy（英文 NER + noun chunks + lemma），对中文和代码路径效果差，
  且引入重型模型依赖；我们用正则级规则，可单测、可审计、零外部依赖。
- 我们的记忆里最有价值的精确信号恰恰是路径 / 工具名 / 包名 / 文件名 ——
  词法通道按 token 拆会丢结构（``a/b/c.py`` 被拆成 a、b、c、py 四个 token，
  反而不如整串匹配精确），向量通道对短实体不敏感。实体层补上这一格。

实体类型（entity_type）：
- PATH        文件/目录路径（Windows/Unix/相对），如 ``C:\\a\\b.py``、``/home/x``、``repo\\src\\main.py``
- FILE        带扩展名的文件名，如 ``main.py``、``config.yml``（在 PATH 内出现的由 PATH 兜走）
- IDENTIFIER  技术标识符：模块链 ``a.b.c``、作用域路径 ``a::b``、蛇形 ``ctx_recall``、
              驼峰类名 ``SqliteLogStore``、仓库路径 ``user/repo``
- QUOTED      引号内容（"..."、'...'、「...」、『...』）
- PROPER      首字母大写的专名词组（Mem0、DeepSeek）

纯函数入口：``extract_entities(text)``；索引：``EntityIndex``（内存倒排 + JSON 持久化）。
"""

from __future__ import annotations

import json
import re
import threading
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple

# 实体类型常量
TYPE_PATH = "PATH"
TYPE_FILE = "FILE"
TYPE_IDENTIFIER = "IDENTIFIER"
TYPE_QUOTED = "QUOTED"
TYPE_PROPER = "PROPER"

# 常见文件扩展名（用于 FILE 类型；新扩展名只需加在这里）
_FILE_EXTS = {
    "py", "pyw", "pyi", "ipynb", "ts", "tsx", "js", "jsx", "mjs", "cjs", "mts", "cts",
    "md", "markdown", "rst", "yml", "yaml", "json", "toml", "ini", "cfg", "conf",
    "txt", "tex", "bib", "sql", "sh", "ps1", "bat", "cmd", "lock", "c", "h", "cc",
    "cpp", "hpp", "cxx", "rs", "go", "java", "kt", "swift", "scala", "lua", "rb",
    "php", "cs", "vue", "svelte", "html", "htm", "css", "scss", "less", "xml",
    "png", "jpg", "jpeg", "gif", "webp", "svg", "ico", "bmp", "tiff", "avif",
    "wav", "mp3", "flac", "ogg", "mp4", "mkv", "mov", "webm", "gif",
    "db", "sqlite", "sqlite3", "zip", "tar", "gz", "xz", "bz2", "7z", "rar",
    "pdf", "doc", "docx", "xls", "xlsx", "ppt", "pptx", "log", "env", "gitignore",
    "npy", "npz", "pt", "pth", "onnx", "safetensors", "pb", "pbtxt", "cfg",
}

# 路径特征：Windows 盘符、Unix 根、相对路径的 ../ ./、含目录分隔符的片段
_PATH_PATTERN = re.compile(
    r"(?:[A-Za-z]:[\\/]|(?:\.{1,2})?[\\/])"
    r"[^\s\"'<>|]+"
)
# 补充：不含盘符/点前缀的“多级目录 + 文件”路径（如 repo\src\main.py、a/b/c.py）
_RELPATH_PATTERN = re.compile(
    r"(?:[\w.\-\u4e00-\u9fff]+[\\/])+[\w.\-\u4e00-\u9fff]+(?:\.[A-Za-z0-9]{1,12})?"
)

# 文件名：`name.ext`，且不在路径片段里（由调用方过滤）
_FILE_PATTERN = re.compile(
    r"(?<![\\/\w.])"  # 前面不是路径分隔符/词字符/点 → 独立的文件名
    r"([\w.\-\u4e00-\u9fff]+)\.([A-Za-z0-9]{1,12})"
    r"(?![\\/\w.])"
)

# 模块链 / 包链：a.b.c
_MODULE_PATTERN = re.compile(r"\b[A-Za-z_][\w-]*(?:\.[A-Za-z_][\w-]*){2,}\b")
# 作用域路径：a::b::c
_SCOPE_PATTERN = re.compile(r"\b[A-Za-z_][\w-]*(?:::[A-Za-z_][\w-]*){1,}\b")
# 仓库路径：user/repo（排除已是路径的）
_REPO_PATTERN = re.compile(r"(?<![\\/\w])[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+(?![\\/\w])")
# 蛇形标识符：ctx_recall（至少一个下划线且不以路径尾形式出现）
_SNAKE_PATTERN = re.compile(r"\b[a-z][a-z0-9]*(?:_[a-z0-9]+){1,}\b")
# 驼峰类名：SqliteLogStore（至少两次大写分段）
_CAMEL_PATTERN = re.compile(r"\b[A-Z][a-z0-9]*(?:[A-Z][a-z0-9]*){1,}\b")

# 引号内容（成对引号/书名号）
_QUOTED_PATTERN = re.compile(
    r"[\"\u201c\u201d'`\u2018\u2019\u300c\u300d\u300e\u300f]"
    r"([^\"'\u201c\u201d\u2018\u2019\u300c\u300d\u300e\u300f]{2,80})"
    r"[\"\u201c\u201d'`\u2018\u2019\u300c\u300d\u300e\u300f]"
)

# 大写专名词组（英文/拼音式）：Mem0 这类混合串交给 CAMEL；这里只收“空格分隔的大写词”
_PROPER_PATTERN = re.compile(r"\b(?:[A-Z][a-z]{1,20}\s+)*[A-Z][a-z]{1,20}\b")

# 通用词过滤（借鉴 mem0 的 _GENERIC_HEADS / _GENERIC_ENDINGS，中英混合版）。
# 命中的实体整体丢弃 —— 这些都是“描述性词头”，不是该记住的实体。
_GENERIC_TERMS = {
    # 英文
    "thing", "stuff", "way", "time", "experience", "situation", "case", "fact",
    "matter", "issue", "idea", "thought", "feeling", "place", "area", "part",
    "kind", "type", "sort", "lot", "bit", "day", "week", "month", "moment",
    "instance", "example", "technique", "method", "approach", "process", "step",
    "tool", "result", "outcome", "goal", "task", "item", "topic", "scale", "size",
    "level", "degree", "amount", "number", "style", "look", "color", "colour",
    "shape", "form", "piece", "section", "side", "end", "edge", "surface", "point",
    "work", "works", "job", "job", "info", "information", "details", "data",
    "content", "material", "materials", "activity", "activities", "effort",
    "efforts", "option", "options", "choice", "choices", "output", "outputs",
    "product", "products", "items", "user", "assistant", "agent", "customer",
    "client", "person", "people", "human", "memory", "message", "conversation",
    "chat", "session", "system", "top", "main", "page", "code", "function",
    "class", "file", "files", "dir", "dirs", "path", "paths", "name", "names",
    "value", "values", "key", "keys", "mode", "type", "types",
    # 中文
    "方法", "方式", "办法", "结果", "结论", "任务", "问题", "内容", "情况", "事情",
    "东西", "地方", "时候", "时间", "步骤", "过程", "流程", "工具", "文件", "数据",
    "信息", "文档", "目录", "路径", "项目", "仓库", "版本", "代码", "函数", "类",
    "接口", "模块", "配置", "参数", "输入", "输出", "返回", "系统", "用户", "助手",
    "会话", "对话", "消息", "答案", "方案", "原因", "效果", "优点", "缺点", "细节",
    "说明", "描述", "命令", "操作", "建议", "记录", "备注", "页面", "功能", "属性",
    "状态", "类型", "名称", "值", "键", "默认", "示例", "例子", "设计", "架构",
    "实现", "部署", "测试", "需求", "目标", "进度", "风险", "清单", "笔记", "文档",
}

# 长度 / 结构过滤
_MIN_ENTITY_LEN = 3          # 实体最短 3 字符（中文 2 字其实有用，QUOTED 单独放宽）
_MAX_ENTITY_LEN = 260        # 与路径上限一致

# 缩略词等“看起来像实体但几乎必然是噪声”的单 token（首字母大写的通用词）
_GENERIC_PROPER = {
    "The", "This", "That", "These", "Those", "And", "Or", "But", "For", "With",
    "From", "Into", "About", "After", "Before", "When", "Where", "How", "Why",
    "What", "Which", "Who", "Not", "All", "Any", "None", "Some", "More", "Less",
    "Very", "Also", "Only", "Just", "Still", "Even", "Well", "Now", "Here",
    "There", "Our", "Your", "My", "His", "Her", "Its", "Their", "First", "Last",
    "Next", "Previous", "Final", "Initial", "Main", "Important", "Key", "New",
    "Old", "Good", "Bad", "High", "Low", "Big", "Small", "Large", "Long", "Short",
    "Please", "Note", "Update", "Fixed", "Issue", "Todo", "FIXME", "TODO", "XXX",
    "Warn", "Error", "Info", "Debug", "Trace", "Status", "Result", "Output",
}

# 路径里常见但无检索价值的段（系统目录/虚拟环境/编译产物）
_PATH_NOISE_PARTS = {
    "node_modules", "venv", ".venv", "env", "__pycache__", ".git", ".idea", ".vscode",
    "dist", "build", "target", "out", "bin", "obj", "site-packages", "lib", "lib64",
    "include", "share", "cache", "logs", "tmp", "temp", ".cache", ".npm", ".pnpm-store",
}


def _norm(text: str) -> str:
    """实体归一化键：小写 + 折叠空白（路径也小写 —— 检索时大小写不敏感）。"""
    return " ".join(text.lower().split())


def _clean(text: str) -> str:
    """清洗：去首尾标点/空白、折叠空白。"""
    return " ".join(text.strip().strip("\\/.'\"`,;:()[]{}<>").split())


def _is_noise(text: str) -> bool:
    t = text.lower()
    if len(t) < _MIN_ENTITY_LEN or len(t) > _MAX_ENTITY_LEN:
        return True
    if t in _GENERIC_TERMS:
        return True
    if any(c in t for c in ("*", "#", "|", "…", "\n", "\t", "  ")):
        return True
    return False


def _path_is_noise(path: str) -> bool:
    low = path.lower().replace("\\", "/")
    parts = [p for p in low.split("/") if p]
    if not parts:
        return True
    # 纯分隔符或纯点
    if set(path) <= {".", "\\", "/"}:
        return True
    # 命中系统噪声段 → 丢掉（比如整个 node_modules 下的绝对路径通常无检索价值）
    if any(p in _PATH_NOISE_PARTS for p in parts):
        return True
    return False


def extract_entities(text: str) -> List[Tuple[str, str]]:
    """从文本中确定性抽取实体，返回 [(entity_type, entity_text), ...]。

    纯函数、无状态、可单测。抽取顺序与去重规则见下。
    """
    if not text:
        return []
    # 引号内容先行抽走（占位掉，避免内嵌的路径/标识符再被重复抽取）
    quoted: List[str] = []
    working = text
    def _drop_quoted(m: re.Match[str]) -> str:
        inner = _clean(m.group(1))
        if not _is_noise(inner):
            quoted.append(inner)
        # 用空格占位，断开内部可能匹配的路径/标识符
        return " " * len(m.group(0))
    working = _QUOTED_PATTERN.sub(_drop_quoted, working)

    found: List[Tuple[str, str, int]] = []  # (type, text, priority)
    seen_spans: List[Tuple[int, int, str]] = []

    # 1) PATH：Windows 盘符/根/相对路径（长路径优先，避免 FILE/IDENTIFIER 抢走子串）
    for m in _PATH_PATTERN.finditer(working):
        cand = _clean(m.group(0))
        if not cand or _path_is_noise(cand):
            continue
        found.append((TYPE_PATH, cand, 0))
        seen_spans.append((m.start(), m.end(), TYPE_PATH))
    for m in _RELPATH_PATTERN.finditer(working):
        cand = _clean(m.group(0))
        if not cand or _path_is_noise(cand):
            continue
        found.append((TYPE_PATH, cand, 0))
        seen_spans.append((m.start(), m.end(), TYPE_PATH))

    def _in_path(start: int, end: int) -> bool:
        return any(s < end and start < e for s, e, _t in seen_spans)

    # 2) IDENTIFIER 家族（模块链 > 作用域 > 仓库路径 > 驼峰 > 蛇形，优先级递减）
    for m in _MODULE_PATTERN.finditer(working):
        if _in_path(m.start(), m.end()):
            continue
        cand = _clean(m.group(0))
        if not _is_noise(cand):
            found.append((TYPE_IDENTIFIER, cand, 1))
    for m in _SCOPE_PATTERN.finditer(working):
        if _in_path(m.start(), m.end()):
            continue
        cand = _clean(m.group(0))
        if not _is_noise(cand):
            found.append((TYPE_IDENTIFIER, cand, 1))
    for m in _REPO_PATTERN.finditer(working):
        if _in_path(m.start(), m.end()):
            continue
        cand = _clean(m.group(0))
        if not _is_noise(cand) and "/" in cand:
            found.append((TYPE_IDENTIFIER, cand, 1))
    for m in _CAMEL_PATTERN.finditer(working):
        if _in_path(m.start(), m.end()):
            continue
        cand = _clean(m.group(0))
        if cand.lower() in _GENERIC_PROPER or _is_noise(cand):
            continue
        found.append((TYPE_IDENTIFIER, cand, 1))
    for m in _SNAKE_PATTERN.finditer(working):
        if _in_path(m.start(), m.end()):
            continue
        cand = _clean(m.group(0))
        if cand.lower() in _GENERIC_PROPER or _is_noise(cand):
            continue
        found.append((TYPE_IDENTIFIER, cand, 2))

    # 3) FILE：独立文件名（不在路径内）
    for m in _FILE_PATTERN.finditer(working):
        if _in_path(m.start(), m.end()):
            continue
        name, ext = m.group(1), m.group(2)
        if ext.lower() not in _FILE_EXTS:
            continue
        cand = _clean(m.group(0))
        if not _is_noise(cand):
            found.append((TYPE_FILE, cand, 2))

    # 4) PROPER：大写专名词组（不在路径/标识符内，且非通用词）
    for m in _PROPER_PATTERN.finditer(working):
        if _in_path(m.start(), m.end()):
            continue
        cand = _clean(m.group(0))
        if not cand or " " not in cand:
            continue  # 单个大写词不构成词组，交给 IDENTIFIER 判定
        low = cand.lower()
        if low in _GENERIC_TERMS or any(w.lower() in _GENERIC_PROPER for w in cand.split()):
            continue
        if _is_noise(cand):
            continue
        found.append((TYPE_PROPER, cand, 3))

    # 5) QUOTED（来自占位收集）
    for inner in quoted:
        found.append((TYPE_QUOTED, inner, 3))

    return _dedup(found)


def _dedup(found: List[Tuple[str, str, int]]) -> List[Tuple[str, str]]:
    """按归一化文本去重（低优先级不覆盖高优先级）+ 重叠裁剪。"""
    best: Dict[str, Tuple[str, str, int]] = {}
    for etype, text, prio in found:
        key = _norm(text)
        cur = best.get(key)
        if cur is None or prio < cur[2]:
            best[key] = (etype, text, prio)
    # 按优先级排序输出（同类内保持文本稳定）
    ordered = sorted(best.values(), key=lambda x: (x[2], x[1]))
    return [(t, tx) for t, tx, _p in ordered]


class EntityIndex:
    """实体↔记忆倒排索引：内存 dict + JSON 持久化（线程安全）。

    链接：``link(text, memory_id)`` 抽取文本实体并登记到 memory_id 名下。
    查询：``boosts_for(entity_texts)`` 返回 {memory_id: {实体文本集合}}。
    持久化：``save()``/``flush()`` 写 JSON；构造时可传 ``path`` 自动载入。
    """

    def __init__(self, path: Optional[str | Path] = None) -> None:
        self._path = Path(path).expanduser() if path else None
        # key=归一化实体文本 -> {entity_type, text, memory_ids: set}
        self._entities: Dict[str, Dict[str, Any]] = {}
        self._lock = threading.RLock()
        self._dirty = False
        self._last_save = 0.0
        if self._path and self._path.exists():
            self._load(self._path)

    @property
    def path(self) -> Optional[Path]:
        return self._path

    @property
    def count(self) -> int:
        with self._lock:
            return len(self._entities)

    def link(self, text: str, memory_id: str, limit: int = 2000) -> int:
        """抽取 text 的实体并登记到 memory_id；返回新登记的实体数。

        文本过长时截断（与嵌入器一致的口径）。线程安全。
        """
        ents = extract_entities(text[:limit] if text else "")
        if not ents:
            return 0
        added = 0
        with self._lock:
            for etype, et in ents:
                key = _norm(et)
                ent = self._entities.get(key)
                if ent is None:
                    ent = {
                        "entity_type": etype,
                        "text": et,
                        "memory_ids": set(),
                        "last_seen": time.time(),
                    }
                    self._entities[key] = ent
                    added += 1
                if memory_id not in ent["memory_ids"]:
                    ent["memory_ids"].add(memory_id)
                ent["last_seen"] = time.time()
            self._dirty = True
            self._maybe_flush()
        return added

    def unlink(self, memory_id: str) -> int:
        """从所有实体上摘除一个记忆（记录删除/过期时调用）；空实体一并清掉。"""
        removed = 0
        with self._lock:
            empty = []
            for key, ent in self._entities.items():
                if memory_id in ent["memory_ids"]:
                    ent["memory_ids"].discard(memory_id)
                    removed += 1
                if not ent["memory_ids"]:
                    empty.append(key)
            for key in empty:
                self._entities.pop(key, None)
            if removed:
                self._dirty = True
                self._maybe_flush()
        return removed

    def boosts_for(
        self,
        entity_texts: Iterable[Tuple[str, str]],
    ) -> Dict[str, Set[str]]:
        """按查询实体集返回 {memory_id: {命中的实体文本}}。

        entity_texts 是 extract_entities(query) 的结果；内部按归一化键精确匹配。
        """
        keys = [_norm(et) for _t, et in entity_texts]
        if not keys:
            return {}
        out: Dict[str, Set[str]] = {}
        with self._lock:
            for key in keys:
                ent = self._entities.get(key)
                if ent is None:
                    continue
                for mid in ent["memory_ids"]:
                    out.setdefault(mid, set()).add(ent["text"])
        return out

    def extract_for_query(self, query_text: str) -> List[Tuple[str, str]]:
        """查询侧实体抽取（薄封装，方便调用方与索引解耦）。"""
        return extract_entities(query_text[:2000] if query_text else "")

    def flush(self) -> None:
        """立即落盘（幂等；无路径或未脏时是空操作）。"""
        with self._lock:
            if not self._dirty or not self._path:
                return
            self._save_locked(self._path)

    def _maybe_flush(self) -> None:
        """节流落盘：脏且距上次保存 > 2s 才写，避免高频 ingest 反复全量写盘。"""
        if not self._path or not self._dirty:
            return
        now = time.time()
        if now - self._last_save >= 2.0:
            self._save_locked(self._path)

    def _save_locked(self, path: Path) -> None:
        payload = {
            key: {
                "entity_type": ent["entity_type"],
                "text": ent["text"],
                "memory_ids": sorted(ent["memory_ids"]),
                "last_seen": ent["last_seen"],
            }
            for key, ent in self._entities.items()
        }
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(path.suffix + ".tmp")
        with tmp.open("w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=1, sort_keys=True)
        tmp.replace(path)
        self._dirty = False
        self._last_save = time.time()

    def _load(self, path: Path) -> None:
        with path.open("r", encoding="utf-8") as f:
            payload = json.load(f)
        for key, ent in payload.items():
            mids = ent.get("memory_ids") or []
            self._entities[key] = {
                "entity_type": ent.get("entity_type", TYPE_IDENTIFIER),
                "text": ent.get("text", key),
                "memory_ids": set(mids),
                "last_seen": ent.get("last_seen", 0.0),
            }

    def to_dict(self) -> Dict[str, Any]:
        with self._lock:
            return {
                key: {
                    "entity_type": ent["entity_type"],
                    "text": ent["text"],
                    "memory_ids": sorted(ent["memory_ids"]),
                }
                for key, ent in self._entities.items()
            }


def entity_raw_hits(
    entity_index: Optional[EntityIndex],
    query_text: str,
    event_ids: Iterable[str],
) -> Dict[str, List[str]]:
    """召回打分共用入口：返回 {event_id: [命中的实体文本]}（无索引/无实体则 {}）。

    三个打分实现（LogVectorStore / SqliteLogStore / fuse_hybrid）都用它，
    保证两后端实体加成语义一致（parity 由测试守护）。
    """
    if entity_index is None or not query_text:
        return {}
    q_ents = entity_index.extract_for_query(query_text)
    if not q_ents:
        return {}
    boosts = entity_index.boosts_for(q_ents)
    ids = list(event_ids)
    return {eid: sorted(boosts.get(eid, ())) for eid in ids}
