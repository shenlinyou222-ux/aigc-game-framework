# -*- coding: utf-8 -*-
"""npc_memory —— AI-NPC 的记忆口子：主动召回 + 推理新向量（门槛与权限以 CEL 规则表达）。

对应 aigc_docs/11（机制主线）与 12（落地设计）。**分工铁律**：
- 触发 = CEL（预设向量相关度过阈值 → 该发起一次推理了）
- 推理 = 大模型（qwen 基于"信条+事实"生成新结论/新事实，程序不替模型下结论）
- 入库 = CEL（self_write_only / grounded / 置信门槛，决定模型结论能不能落成事实向量）
"""
from __future__ import annotations

import json
import math
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

from . import mcel
from .functions import FunctionRegistry, InternalFunction, Rule, RuleError
from .interface_store import HashEmbedder
from .log_vector_store import LogVectorStore
from .models import GameLogRecord

# ============================================================
# 01§6.3 信源置信度模板（基础值表；情境修正由 persona 的 CEL gate 承担）
# ============================================================
SOURCE_CONFIDENCE: Dict[str, float] = {
    "亲眼": 0.95,
    "亲耳": 0.70,
    "同阵营转述": 0.55,
    "敌对阵营转述": 0.15,
    "系统公告": 0.99,
    "推理": 0.55,  # NPC 自己推理出的结论：低于亲历，高于敌对传言
}

DEFAULT_GATE_RECALL = "cand_conf >= threshold"
DEFAULT_GATE_INFERENCE = "(conf >= base_confidence) && (grounded == true)"
DEFAULT_GATE_ACCESS_FACTION = "(cand_faction == npc_faction) || (cand_source == '系统公告')"
DEFAULT_GATE_ACCESS_OPEN = "true"


def load_personas(path):
    """人物数据（npc_personas.json）。文件缺失/为空时返回 {}（全走默认规则）。"""
    p = Path(path)
    if not p.exists():
        return {}
    data = json.loads(p.read_text(encoding="utf-8"))
    return data.get("npcs", data) if isinstance(data, dict) else {}


def confidence_of(record: GameLogRecord) -> float:
    """记录置信度：data.confidence 显式值优先，否则按信源模板，再兜底 0.5。"""
    d = record.data or {}
    if isinstance(d.get("confidence"), (int, float)):
        return float(d["confidence"])
    return SOURCE_CONFIDENCE.get(str(d.get("source_type", "")), 0.5)


def _persona(personas: Dict[str, Any], npc_id: str) -> Dict[str, Any]:
    return personas.get(npc_id, {}) if isinstance(personas, dict) else {}


def cosine(a: List[float], b: List[float]) -> float:
    """纯 Python 余弦（无 numpy 依赖；维度不一致按 0 处理）。"""
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0.0 or nb == 0.0:
        return 0.0
    return dot / (na * nb)


def _embed(embedder: Any, text: str) -> List[float]:
    """兼容两种嵌入器接口：__call__(text) / embed_text(text)。"""
    if embedder is None:
        return []
    if hasattr(embedder, "embed_text"):
        return list(embedder.embed_text(text))
    return list(embedder(text))


# ============================================================
# 主动召回（npc_recall）：阵营访问门 + 记忆门
# ============================================================
def _make_recall_fn(personas: Dict[str, Any], embedder, log_store: LogVectorStore) -> InternalFunction:
    """CEL 前置规则管"能不能查"，access_gate 管"看不看得到"，memory.gate 管"配不配进 prompt"。"""

    def _impl(ctx: Dict[str, Any]) -> Dict[str, Any]:
        persona = _persona(personas, ctx["npc_id"])
        mem = persona.get("memory", {})
        gate = str(mem.get("gate", DEFAULT_GATE_RECALL))
        top_k = int(ctx.get("recall_top_k", mem.get("recall_top_k", 5)))
        max_chars = int(mem.get("max_chars", 900))
        access_gate = str(ctx.get("access_gate", DEFAULT_GATE_ACCESS_OPEN))

        query_text = ctx["query_text"]
        vec = _embed(embedder, query_text)
        hits = log_store.recall_hybrid(
            query_text=query_text,
            query_vector=vec if vec else None,
            limit=max(top_k * 4, 12),
        )
        kept: List[Dict[str, Any]] = []
        dropped = 0
        denied = 0
        for h in hits:
            d = h.record.data or {}
            flat_common = {
                # CEL 变量（扁平）：两道门的表达式只能用这些名字
                "cand_conf": confidence_of(h.record),
                "cand_source": str(d.get("source_type", "")),
                "cand_faction": str(d.get("faction", "")),
                "npc_faction": str(persona.get("faction", "")),
                "threshold": float(ctx.get("confidence_threshold", 0.0)),
                "score": float(h.score),
            }
            # 门1：阵营访问权限（看不看得到这条事实）
            if not mcel.evaluate(access_gate, flat_common):
                denied += 1
                continue
            # 门2：记忆门（配不配进 prompt）
            if mcel.evaluate(gate, flat_common):
                kept.append({
                    "event_id": h.record.event_id,
                    "kind": h.record.kind or h.record.event_type,
                    "text": str(d.get("text", ""))[:160],
                    "confidence": flat_common["cand_conf"],
                    "source_type": flat_common["cand_source"],
                    "faction": flat_common["cand_faction"],
                    "score": flat_common["score"],
                })
            else:
                dropped += 1
        kept = kept[:top_k]

        # 注入前续命（被用即续命，04§8 语义）
        for it in kept:
            try:
                log_store.touch(it["event_id"])
            except Exception:
                pass

        # 记忆块渲染（预算内逐条带置信标签，给 NPC"引述需真实"的依据）
        lines: List[str] = []
        used = 0
        for it in kept:
            row = f"- [{it['source_type'] or '记录'}|置信{it['confidence']:.2f}] {it['text']}"
            if used + len(row) > max_chars:
                break
            lines.append(row)
            used += len(row)
        memory_block = "\n".join(lines) if lines else "（暂无足够可信的记忆）"
        return {
            "memory_block": memory_block,
            "items": kept,
            "gate": gate,
            "access_gate": access_gate,
            "dropped_by_gate": dropped,
            "denied_by_access": denied,
            "recall_top_k": top_k,
        }

    return InternalFunction(
        name="npc_recall",
        description="NPC 主动召回：阵营访问门筛选可见事实，记忆门筛选可信记忆，返回记忆块+候选",
        fn=_impl,
        input_fields={"npc_id": "NPC id（personas 键）", "query_text": "查询原句（轻清洗后）"},
        env_fields=["actor", "persona_ids", "confidence_threshold", "recall_top_k", "access_gate"],
        rules=[
            Rule("npc_known", "npc_id in persona_ids", "未知 NPC，拒绝查询"),
            Rule("actor_allowed",
                 "(actor == npc_id) || (actor == 'system') || (actor == 'assistant')",
                 "只有 NPC 本人/system/assistant 可以发起该 NPC 的召回"),
            Rule("query_not_empty", "size(query_text) >= 2", "查询过短"),
            Rule("threshold_sane", "confidence_threshold >= 0.0 && confidence_threshold <= 1.0",
                 "置信阈值必须在 0~1"),
        ],
    )


# ============================================================
# 预设向量触发（npc_preset_trigger）：只管"要不要发起推理"，不产生结论
# ============================================================
def _make_preset_fn(personas: Dict[str, Any], embedder) -> InternalFunction:
    """预设相关：观察文本 × 每条预设信条做余弦，过 `sim >= threshold`（CEL）即触发推理。"""

    def _impl(ctx: Dict[str, Any]) -> Dict[str, Any]:
        persona = _persona(personas, ctx["npc_id"])
        presets = persona.get("preset_vectors", []) or []
        observed: str = ctx["observed_text"]
        obs_vec = _embed(embedder, observed)
        triggered: List[Dict[str, Any]] = []
        scored: List[Dict[str, Any]] = []
        for p in presets:
            p_vec = _embed(embedder, str(p.get("text", "")))
            sim = cosine(obs_vec, p_vec)
            threshold = float(p.get("threshold", 0.6))
            row = {
                "id": str(p.get("id", "")),
                "text": str(p.get("text", "")),
                "sim": round(sim, 4),
                "threshold": threshold,
                "hint": str(p.get("hint", "")),  # 信条语境(给推理 prompt), 不是结论
            }
            scored.append({"id": row["id"], "text": row["text"],
                           "sim": row["sim"], "threshold": threshold})
            if mcel.evaluate("sim >= threshold", {"sim": sim, "threshold": threshold}):
                triggered.append(row)
        return {"triggered": triggered, "scored": scored, "observed_text": observed}

    return InternalFunction(
        name="npc_preset_trigger",
        description="预设向量触发：观察文本与 NPC 信条余弦相关，过 CEL 阈值=发起一次 LLM 推理",
        fn=_impl,
        input_fields={"npc_id": "NPC id", "observed_text": "观察到的世界事实文本"},
        env_fields=["actor", "persona_ids"],
        rules=[
            Rule("npc_known", "npc_id in persona_ids", "未知 NPC"),
            Rule("observed_not_empty", "size(observed_text) >= 2", "观察文本过短"),
        ],
    )


# ============================================================
# 推理新向量入库（npc_store_inference）：结论由 LLM 产生，这里只做门控与落库
# ============================================================
def _make_inference_fn(personas: Dict[str, Any], log_store: LogVectorStore) -> InternalFunction:
    """CEL 前置规则管"有没有资格写"，persona.inference.gate 管"这条值不值得存"。"""

    def _impl(ctx: Dict[str, Any]) -> Dict[str, Any]:
        persona = _persona(personas, ctx["npc_id"])
        inf = persona.get("inference", {})
        base = float(ctx.get("base_confidence", inf.get("base_confidence",
                        SOURCE_CONFIDENCE["推理"])))
        conf = float(ctx["confidence"]) if ctx.get("confidence") is not None else base
        grounded = bool(ctx.get("grounded", False))
        gate = str(ctx.get("inference_gate", inf.get("gate", DEFAULT_GATE_INFERENCE)))

        flat = {
            "conf": conf,
            "base_confidence": base,
            "grounded": grounded,
            "threshold": float(ctx.get("confidence_threshold", 0.0)),
        }
        if not mcel.evaluate(gate, flat):
            return {"stored": False, "reason": "gate", "gate": gate,
                    "detail": "结论没有事实支撑或置信不达线，不入记忆"}

        npc_id = ctx["npc_id"]
        now = time.time()
        rec = GameLogRecord(
            event_id=f"npc.inference.{uuid.uuid4().hex[:12]}",
            event_type="npc.inference",
            kind="fact",
            session_id=f"npc/{npc_id}",   # NPC 只能落自己的记忆层
            unix_time=now,
            time=str(now),
            data={
                "text": str(ctx["text"]),
                "source_type": "推理",
                "confidence": conf,
                "grounded": grounded,
                "npc": npc_id,
                "faction": str(persona.get("faction", "")),  # 推理事实归 NPC 自己阵营(访问门可见)
                "attitude": ctx.get("attitude", ""),
                "claims": ctx.get("claims", []),
                "evidence": ctx.get("evidence", []),
                "model": ctx.get("model", ""),  # 产生该结论的模型(审计用)
                "extra": ctx.get("extra_data", {}),  # 情绪引擎读数等审计字段
            },
        )
        log_store.add(rec)
        return {
            "stored": True,
            "event_id": rec.event_id,
            "session": rec.session_id,
            "confidence": conf,
            "gate": gate,
        }

    return InternalFunction(
        name="npc_store_inference",
        description="推理新向量入库：LLM 的推理结论过 CEL 门槛后写回 NPC 自己的记忆",
        fn=_impl,
        input_fields={
            "npc_id": "NPC id",
            "text": "结论文本（LLM 推理产出的一句事实性陈述）",
            "confidence": "可选，LLM 自报置信度；缺省用 persona.inference.base_confidence",
            "grounded": "是否有事实支撑（证据事件在库且对得上才为 true）",
            "claims": "可选，支撑该结论的 claims 列表",
        },
        env_fields=["actor", "persona_ids", "base_confidence", "confidence_threshold", "inference_gate"],
        rules=[
            Rule("npc_known", "npc_id in persona_ids", "未知 NPC"),
            Rule("self_write_only",
                 "(actor == npc_id) || (actor == 'system')",
                 "NPC 只能写自己的记忆；他人记忆与永久层仅 system"),
            Rule("text_min", "size(text) >= 4", "结论太短，不值得存"),
            Rule("confidence_sane",
                 "(confidence >= 0.0) && (confidence <= 1.0)",
                 "显式置信度必须在 0~1"),
        ],
    )


# ============================================================
# Prompt 组装
# ============================================================
PROMPT_TEMPLATE = """\
[{display_name}] 角色
{role_line}
说话风格：{tone}，口头禅「{catchphrase}」，每次最多 {max_words} 字。
性格：{personality}

你记忆中与玩家有关的事（引述必须真实，没有就别说）：
{memory_block}

玩家说：{player_text}

请用你的口吻回 1~2 句话。只输出 JSON（不要多余文本）：
{{
  "lines": ["..."],
  "emotion": "表情关键词",
  "trigger": {{"type": "proximity", "radius_px": 64}},
  "claims": [{{"text": "你陈述的事实", "event_type": "对应事件类型", "day": 1}}],
  "new_memory": {{"text": "这轮让你记住的一句新结论", "grounded": true, "confidence": 0.8}}
}}
要求：
- lines 中的每一条事实性陈述都必须在 claims 里登记；没有把握就不要陈述事实；
- 记忆块里没有的事不许编（claim_checker 会逐条对账，CONTRADICTED=幻觉实锤）；
- trigger 是这条对话的触发条件（proximity=玩家进入 radius_px 像素范围时触发）。
"""

# 推理专用 prompt（更小, 给小模型）：信条+事实 → 态度结论（结论由模型产出）
INFERENCE_PROMPT_TEMPLATE = """\
你是游戏里的 NPC「{display_name}」。
你的信条：{preset_text}（这条信条此刻被触发了，相关度 {sim:.2f}）
刚发生的事（有据可查）：{evidence_text}
你原本对玩家的态度：{old_attitude}

请推理：这件事让你对玩家的态度变成了什么？只输出一行 JSON：
{{"attitude": "两字态库名词", "new_memory": "一句完整事实结论（含你对玩家的态度）", "confidence": 0.85}}
示例：{{"attitude": "恐惧", "new_memory": "老鼠B认为玩家会伤害同伴，对玩家充满恐惧", "confidence": 0.9}}
要求：new_memory 必须以"老鼠B"开头、把你的态度写进结论里；结论必须由"刚发生的事"支撑，不许引入没发生的事；confidence 在 0~1 之间。
"""


def build_npc_prompt(persona: Dict[str, Any], memory_block: str, player_text: str) -> str:
    speech = persona.get("speech", {})
    return PROMPT_TEMPLATE.format(
        display_name=persona.get("display_name", persona.get("node", "NPC")),
        role_line=persona.get("role_line", "你是村里的居民。"),
        tone=speech.get("tone", "自然"),
        catchphrase=speech.get("catchphrase", ""),
        max_words=speech.get("max_words", 25),
        personality="、".join(persona.get("personality", [])) or "普通村民",
        memory_block=memory_block or "（暂无）",
        player_text=player_text,
    )


# ---------------------------------------------------------------- 对话 mini 契约（任务拆分版）
# 模型只负责：trigger（开头一个类型词，显式给）+ lines + emotion。
# claims 对账归管线（claim_checker），new_memory 归推理卷 LoRA。
# trigger 路由口子：新触发类型在 TRIGGER_ROUTES 注册即可；模型不给/给不认识的类型 → 默认玩家交互触发。
TRIGGER_ROUTES: Dict[str, Dict[str, Any]] = {
    "proximity": {"type": "proximity", "radius_px": 64},
}
DEFAULT_TRIGGER = "proximity"


def normalize_trigger(t: Any) -> Dict[str, Any]:
    """模型输出里的 trigger 字段 → 触发参数路由。不给/不合法 → 默认玩家交互触发。"""
    if isinstance(t, dict):  # 容忍旧结构 {"type": ...}
        t = t.get("type")
    if isinstance(t, str) and t.strip() and t.strip() in TRIGGER_ROUTES:
        return dict(TRIGGER_ROUTES[t.strip()])
    return dict(TRIGGER_ROUTES[DEFAULT_TRIGGER])


DIALOGUE_MINI_TEMPLATE = """\
[{display_name}] 角色
{role_line}
说话风格：{tone}，口头禅「{catchphrase}」，每次最多 {max_words} 字。
性格：{personality}

你记忆中与玩家有关的事（引述必须真实，没有就别说）：
{memory_block}

玩家说：{player_text}

请用你的口吻回 1~2 句话。只输出一行 JSON（不要多余文本）：
{{"trigger": "proximity", "lines": ["..."], "emotion": "表情关键词"}}
要求：
- trigger 填这轮对话的触发类型：玩家交互触发填 "proximity"（大多数情况都是它），确实是别的触发就填对应类型词；
- 台词里别陈述记忆块里没有的事实（claim_checker 会逐条对账，编造=幻觉实锤）。
"""

# mini prompt 与旧 prompt 的公共前缀切分标记（数据派生用）
DIALOGUE_PROMPT_TAIL_MARKER = "请用你的口吻回"


def build_npc_prompt_mini(persona: Dict[str, Any], memory_block: str, player_text: str) -> str:
    speech = persona.get("speech", {})
    return DIALOGUE_MINI_TEMPLATE.format(
        display_name=persona.get("display_name", persona.get("node", "NPC")),
        role_line=persona.get("role_line", "你是村里的居民。"),
        tone=speech.get("tone", "自然"),
        catchphrase=speech.get("catchphrase", ""),
        max_words=speech.get("max_words", 25),
        personality="、".join(persona.get("personality", [])) or "普通村民",
        memory_block=memory_block or "（暂无）",
        player_text=player_text,
    )


def build_inference_prompt(
    persona: Dict[str, Any],
    preset: Dict[str, Any],
    evidence_text: str,
    old_attitude: str = "中性",
) -> str:
    """推理 prompt：把"信条+事实"交给模型，让模型自己产出态度结论。"""
    return INFERENCE_PROMPT_TEMPLATE.format(
        display_name=persona.get("display_name", persona.get("node", "NPC")),
        preset_text=preset.get("text", ""),
        sim=float(preset.get("sim", 0.0)),
        evidence_text=evidence_text,
        old_attitude=old_attitude,
    )


# ============================================================
# 组装入口
# ============================================================
class NpcMemory:
    """把人物数据 + 记忆库组装成带 CEL 规则的内部函数组。"""

    def __init__(
        self,
        log_store: LogVectorStore,
        personas: Dict[str, Any],
        embedder: Optional[Any] = None,
        actor: str = "system",
    ) -> None:
        self.log_store = log_store
        self.personas = personas or {}
        self.embedder = embedder or HashEmbedder()
        self.actor = actor
        self._recall = _make_recall_fn(self.personas, self.embedder, log_store)
        self._preset = _make_preset_fn(self.personas, self.embedder)
        self._infer = _make_inference_fn(self.personas, log_store)

    # -- env 组装：把 persona 的数值/表达式拍平成 CEL 上下文 --
    def _env(self, npc_id: str, actor: Optional[str] = None) -> Dict[str, Any]:
        persona = _persona(self.personas, npc_id)
        mem = persona.get("memory", {})
        inf = persona.get("inference", {})
        access_default = (DEFAULT_GATE_ACCESS_FACTION if persona.get("faction")
                          else DEFAULT_GATE_ACCESS_OPEN)
        return {
            "actor": actor or self.actor,
            "persona_ids": sorted(self.personas.keys()),
            "confidence_threshold": float(mem.get("confidence_threshold", 0.0)),
            "recall_top_k": int(mem.get("recall_top_k", 5)),
            "access_gate": str(persona.get("access_gate", access_default)),
            "base_confidence": float(inf.get("base_confidence", SOURCE_CONFIDENCE["推理"])),
            "inference_gate": str(inf.get("gate", DEFAULT_GATE_INFERENCE)),
        }

    # ---- 世界事实入库（如"玩家杀死了老鼠A"）----
    def ingest_world_fact(
        self,
        text: str,
        *,
        source_type: str = "系统公告",
        faction: str = "",
        event_type: str = "world.fact",
        session_id: str = "world",
        extra: Optional[Dict[str, Any]] = None,
    ) -> str:
        data = {
            "text": text,
            "source_type": source_type,
            "confidence": SOURCE_CONFIDENCE.get(source_type, 0.5),
            "faction": faction,
        }
        if extra:
            data.update(extra)
        rec = GameLogRecord(
            event_id=f"{event_type}.{uuid.uuid4().hex[:12]}",
            event_type=event_type,
            kind="event",
            session_id=session_id,
            unix_time=time.time(),
            time=str(time.time()),
            data=data,
        )
        self.log_store.add(rec)
        return rec.event_id

    # ---- 主动召回 ----
    def recall(self, npc_id: str, query_text: str, actor: Optional[str] = None) -> Dict[str, Any]:
        return self._recall.call({"npc_id": npc_id, "query_text": query_text},
                                 self._env(npc_id, actor))

    # ---- 预设向量触发（只判断"要不要推理"）----
    def preset_triggers(self, npc_id: str, observed_text: str) -> Dict[str, Any]:
        return self._preset.call({"npc_id": npc_id, "observed_text": observed_text},
                                 self._env(npc_id))

    def store_inference(
        self,
        npc_id: str,
        text: str,
        grounded: bool = False,
        confidence: Optional[float] = None,
        claims: Optional[List[Dict[str, Any]]] = None,
        evidence: Optional[List[str]] = None,
        attitude: str = "",
        model: str = "",
        actor: Optional[str] = None,
        extra_data: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """把**大模型产出的推理结论**过 CEL 门槛后入库（text 必须来自模型，不是程序拼的）。"""
        return self._infer.call(
            {
                "npc_id": npc_id,
                "text": text,
                "grounded": grounded,
                "confidence": confidence,
                "claims": claims or [],
                "evidence": evidence or [],
                "attitude": attitude,
                "model": model,
                "extra_data": extra_data or {},
            },
            self._env(npc_id, actor),
        )

    def prompt(self, npc_id: str, query_text: str, player_text: str) -> Dict[str, Any]:
        """先召回再组 prompt 的便捷入口（返回 prompt + 召回审计信息）。"""
        r = self.recall(npc_id, query_text)
        persona = _persona(self.personas, npc_id)
        return {
            "prompt": build_npc_prompt(persona, r["memory_block"], player_text),
            "recall": r,
        }

    def register(self, registry: Optional[FunctionRegistry] = None) -> FunctionRegistry:
        reg = registry or FunctionRegistry()
        reg.register(self._recall)
        reg.register(self._preset)
        reg.register(self._infer)
        return reg