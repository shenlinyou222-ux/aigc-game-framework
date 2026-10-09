# -*- coding: utf-8 -*-
"""llm_json —— 小模型 JSON 输出的校验/自适配器（qwen3.5:0.8b 专用通道）。

三层防线，每层只补上一层的漏：
  1. 语法修（开源 json_repair）：裸键丢引号/单引号/尾逗号/截断/围栏/前后噪声——
     实测修好 0.8b 真实坏输出 ``{"attitude": "敌意", ,confidence: 0.85}``；
  2. 全角标点修（自有，字符串感知）：只在字符串**外**把 ：，（）换回 ASCII，
     台词/信条值里的中文标点原样保留（json_repair 不认全角结构位）；
  3. schema 修（pydantic）：类型矫正（confidence "0.85"→0.85、"90%"→0.9）、
     形状归一（lines 字符串→列表、new_memory 字符串→对象）、缺省补全（trigger）。

调用侧 ``generate_json()`` 一次拿（对象，审计）：
  - 0.8b 非常驻（2026-09-07 策略反转）：``keep_alive=None``→走 ``model_batch_manager`` 策略
    （45s 空闲窗，批内保暖、空闲自动卸载给训练腾卡；早前"防 5s 冷启改 -1 常驻"的决策被
    动态批处理管理取代，cold/warm 计数见该模块 STATS）；
  - ``num_ctx=2048``（NPC prompt ≤1k token；实测 32k ctx 多吃 0.45GB 显存）；
  - 优先结构化输出（format=JSON schema，语法由服务端文法约束），失败降级 format="json"，
    再失败裸跑——三层修复兜底。
置信度越界走向：适配层只做显而易见的矫正（百分数除 100 是 fail-closed 方向），
0~1 之外的硬约束仍由 npc_memory 的 CEL 规则 confidence_sane 拒收，两层不重复设防。
"""
from __future__ import annotations

import json
import re
import urllib.error
import urllib.request
from typing import Any, Callable, Dict, List, Optional, Tuple

from json_repair import repair_json as _repair_json
from pydantic import BaseModel, Field, field_validator

OLLAMA_BASE = "http://127.0.0.1:11434"
QWEN_MODEL = "qwen3.5:0.8b"

try:
    from model_batch_manager import policy_keep_alive  # 动态批处理管理(2026-09-07): keep_alive 策略唯一入口
except ImportError:  # aigc_core 被单独拷走时兜底, 与管理器默认策略一致
    def policy_keep_alive(model: str = QWEN_MODEL):
        return "45s"

# ------------------------------------------------------------
# 第 1+2 层：语法修 + 全角标点修（字符串感知）
# ------------------------------------------------------------

_THINK_RE = re.compile(r"<think>.*?</think>", re.S | re.I)

# 全角 → ASCII，只允许出现在字符串外（结构位）；字符串内的中文标点是台词的一部分
_FULLWIDTH_MAP = {"：": ":", "，": ",", "（": "(", "）": ")", "｛": "{", "｝": "}"}


def strip_think(text: str) -> str:
    return _THINK_RE.sub("", text or "").strip()


def _fix_fullwidth_outside_strings(text: str) -> str:
    """扫描器：跟踪 in_string/escape 状态，只改写结构位的全角标点。"""
    out: List[str] = []
    in_str = False
    quote = ""
    i = 0
    while i < len(text):
        ch = text[i]
        if in_str:
            out.append(ch)
            if ch == "\\" and i + 1 < len(text):
                out.append(text[i + 1])
                i += 2
                continue
            if ch == quote:
                in_str = False
            i += 1
            continue
        if ch in ("\"", "'"):
            in_str = True
            quote = ch
            out.append(ch)
            i += 1
            continue
        out.append(_FULLWIDTH_MAP.get(ch, ch))
        i += 1
    return "".join(out)


def repair_json(text: str) -> Tuple[Optional[Any], List[str]]:
    """坏文本 → (Python 对象 | None, 修复审计)。三层依序降级，记录实际触发层。"""
    raw = text or ""
    repairs: List[str] = []
    # a. 原样直解（模型表现好时零开销）
    try:
        return json.loads(raw), repairs
    except Exception:
        pass
    # b. think 泄漏剥离
    stripped = strip_think(raw)
    if stripped != raw.strip():
        repairs.append("strip_think")
    try:
        return json.loads(stripped), repairs
    except Exception:
        pass
    # c. 全角结构位修（字符串感知，台词值里的中文标点不动）
    fw = _fix_fullwidth_outside_strings(stripped)
    if fw != stripped:
        repairs.append("fullwidth_punct")
    try:
        return json.loads(fw), repairs
    except Exception:
        pass
    # d. json_repair 兜底（裸键/单引号/尾逗号/截断/围栏/噪声）
    obj = _repair_json(fw, return_objects=True)
    if isinstance(obj, (dict, list)) and obj:
        repairs.append("json_repair")
        return obj, repairs
    return None, repairs + ["unrepairable"]


# ------------------------------------------------------------
# 第 3 层：schema 修（pydantic 类型矫正/形状归一/缺省）
# ------------------------------------------------------------

def _coerce_confidence(v: Any) -> Optional[float]:
    """"0.85"→0.85；"90%"→0.9；90→0.9（百分数方向，fail-closed：小值会被 CEL 门拒）。"""
    if v is None or v == "":
        return None
    if isinstance(v, str):
        v = v.strip().rstrip("%")
        try:
            v = float(v)
        except ValueError:
            return None
    v = float(v)
    if v > 1.0:
        v = v / 100.0
    return round(max(0.0, min(1.0, v)), 4)


class InferenceVerdict(BaseModel):
    """npc_memory 推理结论（npc_scene_mouse_b S4）：attitude/new_memory/confidence。"""

    attitude: str = ""
    new_memory: str = ""
    confidence: Optional[float] = None

    @field_validator("attitude", "new_memory", mode="before")
    @classmethod
    def _s(cls, v: Any) -> str:
        return str(v).strip() if v is not None else ""

    @field_validator("confidence", mode="before")
    @classmethod
    def _c(cls, v: Any) -> Optional[float]:
        return _coerce_confidence(v)


class MemoryItem(BaseModel):
    """对话输出里的 new_memory 子对象；0.8b 常退化成裸字符串，此处归一。"""

    text: str = ""
    grounded: Optional[bool] = None
    confidence: Optional[float] = None

    @field_validator("text", mode="before")
    @classmethod
    def _t(cls, v: Any) -> str:
        return str(v).strip() if v is not None else ""

    @field_validator("confidence", mode="before")
    @classmethod
    def _c(cls, v: Any) -> Optional[float]:
        return _coerce_confidence(v)


class DialogueReply(BaseModel):
    """npc_memory 对话输出（S7）：lines/emotion/trigger/claims/new_memory。"""

    lines: List[str] = Field(default_factory=list)
    emotion: str = ""
    trigger: Dict[str, Any] = Field(default_factory=lambda: {"type": "proximity", "radius_px": 64})
    claims: List[Dict[str, Any]] = Field(default_factory=list)
    new_memory: Optional[MemoryItem] = None

    @field_validator("lines", mode="before")
    @classmethod
    def _l(cls, v: Any) -> List[str]:
        if v is None:
            return []
        if isinstance(v, str):
            # 只按换行/分号断句；台词内的逗号是内容，保留
            v = [p for p in re.split(r"[\n；;]", v) if p.strip()]
        return [str(x).strip() for x in v if str(x).strip()]

    @field_validator("emotion", mode="before")
    @classmethod
    def _e(cls, v: Any) -> str:
        return str(v).strip() if v is not None else ""

    @field_validator("new_memory", mode="before")
    @classmethod
    def _m(cls, v: Any):
        if isinstance(v, str):
            return {"text": v}
        return v


# ------------------------------------------------------------
# Ollama 结构化输出 schema（服务端文法约束，第一道防线）
# ------------------------------------------------------------

INFERENCE_SCHEMA = {
    "type": "object",
    "properties": {
        "attitude": {"type": "string"},
        "new_memory": {"type": "string"},
        "confidence": {"type": "number"},
    },
    "required": ["attitude", "new_memory"],
}

DIALOGUE_SCHEMA = {
    "type": "object",
    "properties": {
        "lines": {"type": "array", "items": {"type": "string"}},
        "emotion": {"type": "string"},
        "trigger": {"type": "object"},
        "claims": {"type": "array"},
        "new_memory": {"type": "object"},
    },
    "required": ["lines"],
}


# ------------------------------------------------------------
# 调用侧：generate_json = 非常驻 qwen(动态批处理策略) + 结构化输出 + 三层修复
# ------------------------------------------------------------

def generate_json(
    prompt: str,
    *,
    adapter: Callable[[Any], Tuple[Optional[BaseModel], Dict[str, Any]]] | None = None,
    model: str = QWEN_MODEL,
    base_url: str = OLLAMA_BASE,
    json_schema: Optional[Dict[str, Any]] = None,
    keep_alive: str | int | None = None,   # None→动态批处理管理策略(qwen3.5 非常驻)；显式传 -1/0/字符串照旧(必须整数, Go duration 拒字符串)
    num_ctx: int = 2048,
    temperature: float = 0.8,
    num_predict: int = 256,
    timeout: int = 120,
    transport: Optional[Callable[[Dict[str, Any]], Dict[str, Any]]] = None,
) -> Tuple[Optional[BaseModel], Dict[str, Any]]:
    """调 Ollama 生成并解析 JSON。返回 (pydantic 对象 | None, 审计)。

    adapter: adapt_inference / adapt_dialogue；None 时跳过 schema 层，返回原始 obj。
    transport: 注入假传输做单测；默认走真实 HTTP。
    """
    audit: Dict[str, Any] = {"model": model, "repairs": [], "format_fallback": False}

    def _post(body: Dict[str, Any]) -> Dict[str, Any]:
        if transport is not None:
            return transport(body)
        req = urllib.request.Request(
            f"{base_url}/api/generate",
            data=json.dumps(body, ensure_ascii=False).encode("utf-8"),
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return json.loads(r.read())

    body = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "think": False,
        "keep_alive": keep_alive if keep_alive is not None else policy_keep_alive(model),
        "options": {"temperature": temperature, "num_predict": num_predict, "num_ctx": num_ctx},
    }
    if json_schema is not None:
        body["format"] = json_schema       # 结构化输出：服务端文法约束
    else:
        body["format"] = "json"
    try:
        resp = _post(body)
    except urllib.error.HTTPError:
        # 与原 ollama_generate 同款降级：去掉 think/format 重试一次
        audit["format_fallback"] = True
        body.pop("think", None)
        body.pop("format", None)
        resp = _post(body)
    raw = str(resp.get("response", ""))
    audit["raw"] = raw
    obj, repairs = repair_json(raw)
    audit["repairs"] = repairs
    if obj is None:
        return None, audit
    if adapter is not None:
        adapted, extra = adapter(obj)
        audit.update(extra)
        return adapted, audit
    return obj, audit


def adapt_inference(obj: Any) -> Tuple[Optional[InferenceVerdict], Dict[str, Any]]:
    """推理结论适配：类型矫正 + 缺字段容错（confidence 缺 → None → 调用方落人物默认值）。"""
    try:
        return InferenceVerdict.model_validate(obj), {}
    except Exception as exc:
        return None, {"adapt_error": str(exc)}


def adapt_dialogue(obj: Any) -> Tuple[Optional[DialogueReply], Dict[str, Any]]:
    """对话适配：lines/new_memory 形状归一 + trigger 缺省（proximity/64px）。"""
    try:
        return DialogueReply.model_validate(obj), {}
    except Exception as exc:
        return None, {"adapt_error": str(exc)}
