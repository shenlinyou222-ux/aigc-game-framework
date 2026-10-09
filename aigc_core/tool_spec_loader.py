"""工具 schema 注册表 → 上下文向量库 同步器。

设计要点：
- 注册表：aigc_demo_framework/tool_registry/*.json（结构化 tool spec）。
- 入库形态：GameLogRecord(event_type="tool/schema", kind="tool/schema", session_id="")
  → 全局长期记忆，任何 session 均可经 /ctx/recall(event_type="tool/schema") 召回，
  也可经 /ctx/tool/search 召回。
- 幂等/版本化：event_id = toolspec/<tool>[:<command>]@<sha1(text)[:10]>，
  内容不变 → 同 id 覆盖（无重复行）；内容变更 → 新 id 入库，旧 id 随生命周期过期。
- 寿命：入库后设置 duration（默认 90 天）；被召回命中会 touch 续命。

双路径：
- sync_registry_http(...)      对运行中的 ctx 后端 HTTP 同步（推荐，服务端负责嵌入）。
- sync_registry_into_controller(...)  进程内直写（api.py 启动钩子用，同一 ingest 代码）。
"""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional
from urllib import error as urllib_error
from urllib import request as urllib_request

REGISTRY_EVENT_TYPE = "tool/schema"
DEFAULT_DURATION_SEC = 90 * 24 * 3600  # 90 天；命中即 touch 续命


def _hash10(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:10]


def _record_text(tool: str, title: str, body: str) -> str:
    """召回文本：前 300 字符是 ctx_recall snippet 的窗口，关键信息前置。"""
    return f"【tool/schema】{tool} · {title}\n{body}".strip()


def load_specs(registry_dir: Path) -> List[Dict[str, Any]]:
    specs: List[Dict[str, Any]] = []
    for path in sorted(Path(registry_dir).glob("*.json")):
        try:
            spec = json.loads(path.read_text(encoding="utf-8"))
        except Exception as exc:  # noqa: BLE001 - 单文件损坏不拖垮整体
            print(f"[tool_registry] skip {path.name}: {exc}")
            continue
        if isinstance(spec, dict) and spec.get("tool"):
            spec["_source_file"] = path.name
            specs.append(spec)
    return specs


def spec_to_records(spec: Dict[str, Any], now: Optional[float] = None) -> List["Dict[str, Any]"]:
    """一个 spec 拆成多条记录：overview 一条 + 每个子命令一条（召回粒度更准）。"""
    now = now if now is not None else time.time()
    tool = str(spec["tool"])
    title = str(spec.get("title", tool))
    bin_ref = str(spec.get("bin_path") or spec.get("bin") or "")
    category = str(spec.get("category", ""))

    when = "\n".join(f"- {w}" for w in spec.get("when_to_use", []))
    summary = str(spec.get("summary", "")).strip()
    keywords = " ".join(str(k) for k in spec.get("keywords", []))
    params = spec.get("parameters")
    output = spec.get("output")
    params_text = (
        f"\n参数(schema): {json.dumps(params, ensure_ascii=False)}" if params is not None else ""
    )
    output_text = (
        f"\n输出(schema): {json.dumps(output, ensure_ascii=False)}" if output is not None else ""
    )
    overview_text = _record_text(
        tool,
        title,
        f"用途: {summary}\n可执行: {bin_ref}\n类别: {category}\n关键词: {keywords}\n适用场景:\n{when}"
        + params_text
        + output_text,
    )
    records: List[Dict[str, Any]] = [
        {
            "event_id": f"toolspec/{tool}:overview@{_hash10(overview_text)}",
            "event_type": REGISTRY_EVENT_TYPE,
            "kind": REGISTRY_EVENT_TYPE,
            "session_id": "",  # 空 = 全局长期记忆
            "unix_time": now,
            "data": {
                "text": overview_text,
                "tool": tool,
                "title": title,
                "bin": bin_ref,
                "category": category,
                "command": "",
                "source_file": spec.get("_source_file", ""),
            },
        }
    ]

    for cmd in spec.get("commands", []):
        name = str(cmd.get("name", "")).strip()
        usage = str(cmd.get("usage", "")).strip()
        desc = str(cmd.get("desc", "")).strip()
        if not name:
            continue
        text = _record_text(
            tool,
            title,
            f"子命令 {tool} {name}\n用法: {usage}\n说明: {desc}\n可执行: {bin_ref}",
        )
        records.append(
            {
                "event_id": f"toolspec/{tool}:{name}@{_hash10(text)}",
                "event_type": REGISTRY_EVENT_TYPE,
                "kind": REGISTRY_EVENT_TYPE,
                "session_id": "",
                "unix_time": now,
                "data": {
                    "text": text,
                    "tool": tool,
                    "title": title,
                    "bin": bin_ref,
                    "category": category,
                    "command": name,
                    "usage": usage,
                    "source_file": spec.get("_source_file", ""),
                },
            }
        )
    return records


def registry_to_records(registry_dir: Path) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for spec in load_specs(registry_dir):
        out.extend(spec_to_records(spec))
    return out


def _post_json(base_url: str, path: str, payload: Dict[str, Any], timeout: float = 8.0) -> Dict[str, Any]:
    req = urllib_request.Request(
        f"{base_url.rstrip('/')}{path}",
        data=json.dumps(payload).encode("utf-8"),
        headers={"content-type": "application/json"},
        method="POST",
    )
    with urllib_request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def sync_registry_http(
    registry_dir: Path,
    base_url: str = "http://127.0.0.1:8000",
    duration_sec: float = DEFAULT_DURATION_SEC,
) -> List[str]:
    """对运行中的后端：/ctx/ingest 逐条入库 + /ctx/lifecycle/duration 设寿命。"""
    synced: List[str] = []
    for record in registry_to_records(registry_dir):
        result = _post_json(base_url, "/ctx/ingest", {"record": record})
        if not result.get("ok"):
            raise RuntimeError(f"ingest failed for {record['event_id']}: {result}")
        try:
            _post_json(
                base_url,
                "/ctx/lifecycle/duration",
                {"event_id": record["event_id"], "duration_sec": duration_sec},
            )
        except urllib_error.URLError:
            pass  # 设寿命失败不影响入库
        synced.append(record["event_id"])
    return synced


def sync_registry_into_controller(controller: Any, registry_dir: Path) -> List[str]:
    """进程内直写（api.py 启动钩子）：走与 HTTP /ctx/ingest 相同的 context_ingest。"""
    from .models import GameLogRecord

    synced: List[str] = []
    for record in registry_to_records(registry_dir):
        controller.context_ingest(GameLogRecord(**record))
        synced.append(record["event_id"])
    return synced
