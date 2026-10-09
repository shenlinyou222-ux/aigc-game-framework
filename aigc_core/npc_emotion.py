# -*- coding: utf-8 -*-
"""npc_emotion —— Gamygdala 式情绪引擎（迷你移植版）：给 NPC 量化的情绪系数。

抄作业对象：Gamygdala (Broekens & van Gemert, TU Delft) —— 评价理论(appraisal)的
轻量游戏情绪引擎。核心量化公式：

    情绪强度 intensity = |utility × delta| × likelihood     （目标效用×事件增量×达成概率）
    方向：delta>0 目标一致(joy/hope) ；delta<0 目标违背(distress/fear)
    归因：负面事件由他人引起 → anger 指向对方；反复 anger + like<0 → hate（衰减更慢）
          好事由他人引起 → gratitude
    关系：like ∈ [-1,1] 随归因持久更新（like += sign(desirability)×|intensity|×gain）
    衰减：每拍 intensity ×= (1 - decay_rate)（hate 乘 hate_decay_factor 衰得更慢）
    态度：attitude = clamp(like + 0.5×gratitude - 0.5×(anger+hate)) → 词表映射

我们的接法：引擎产出**数值系数**（anger/hate/like/attitude_score），LLM 只负责把这些
系数"翻译"成结论与台词；推理事实入库时带引擎读数（可审计，不靠模型自报唯一来源）。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


# ------------------------------------------------------------
# 数据结构
# ------------------------------------------------------------
@dataclass
class Goal:
    name: str
    utility: float            # 效用 [-1,1]：越在乎越大（负效用=回避目标）
    likelihood: float = 1.0   # 达成概率 [0,1]
    is_incremental: bool = False


@dataclass
class Emotion:
    name: str
    intensity: float = 0.0
    decay_factor: float = 1.0  # hate 等慢衰减情绪 <1


@dataclass
class Relation:
    like: float = 0.0          # [-1,1] 对另一 agent 的持久好感


@dataclass
class Agent:
    name: str
    faction: str = ""
    goals: List[Goal] = field(default_factory=list)
    relations: Dict[str, Relation] = field(default_factory=dict)   # other -> Relation
    emotions: Dict[str, Emotion] = field(default_factory=dict)     # label -> Emotion

    def relation(self, other: str) -> Relation:
        if other not in self.relations:
            self.relations[other] = Relation()
        return self.relations[other]

    def add_emotion(self, name: str, intensity: float, decay_factor: float = 1.0) -> None:
        cur = self.emotions.get(name)
        if cur is None:
            self.emotions[name] = Emotion(name=name, intensity=min(intensity, 1.0),
                                          decay_factor=decay_factor)
        else:
            cur.intensity = min(cur.intensity + intensity, 1.0)
            cur.decay_factor = decay_factor


# ------------------------------------------------------------
# 引擎
# ------------------------------------------------------------
DEFAULT_ENGINE_CONFIG = {
    "decay_per_day": 0.30,       # 每游戏日指数衰减率
    "hate_decay_factor": 0.5,    # hate 衰减率 = decay_per_day × factor（更持久）
    "attribution_gain": 1.0,     # 归因对 like 的更新增益
    "attitude_gain": 0.5,        # 态度合成时累积情绪的权重
}

ATTITUDE_WORDS = [
    (-1.00, -0.60, "仇恨"),
    (-0.60, -0.25, "敌意"),
    (-0.25, 0.25, "中性"),
    (0.25, 0.60, "友善"),
    (0.60, 1.01, "挚友"),
]


def clamp(x: float, lo: float = -1.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, x))


def attitude_word(score: float) -> str:
    for lo, hi, word in ATTITUDE_WORDS:
        if lo <= score < hi:
            return word
    return "中性"


class EmotionEngine:
    """Gamygdala 式评价引擎。config 来自 persona.emotion_engine（数据可调）。"""

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        cfg = dict(DEFAULT_ENGINE_CONFIG)
        cfg.update(config or {})
        self.config = cfg

    # ---- 评价：一条世界事件 → 该 agent 的情绪/关系更新 ----
    def appraise(
        self,
        agent: Agent,
        event: Dict[str, Any],
        self_name: str,
    ) -> Dict[str, Any]:
        """event: {causal, target, likelihood, goals:{goal_name: delta}}"""
        causal = str(event.get("causal", ""))
        likelihood = float(event.get("likelihood", 1.0))
        deltas: Dict[str, float] = event.get("goals", {})
        gain = float(self.config["attribution_gain"])

        produced: List[Dict[str, Any]] = []
        total_desirability = 0.0
        for goal in agent.goals:
            delta = float(deltas.get(goal.name, 0.0))
            if delta == 0.0:
                continue
            desirability = clamp(goal.utility * delta)
            total_desirability += desirability
            intensity = min(abs(desirability) * likelihood, 1.0)
            pending = goal.likelihood < 1.0 and not goal.is_incremental
            # 自身情绪
            if desirability >= 0:
                label = "hope" if pending else "joy"
            else:
                label = "fear" if pending else "distress"
            agent.add_emotion(label, intensity)
            produced.append({"goal": goal.name, "delta": delta,
                             "desirability": round(desirability, 3),
                             "emotion": label, "intensity": round(intensity, 3)})

        # 归因情绪（因果人是别人时）：anger / hate / gratitude + like 更新
        if causal and causal != self_name and total_desirability != 0.0:
            rel = agent.relation(causal)
            intensity = min(abs(total_desirability) * likelihood, 1.0)
            if total_desirability < 0:
                agent.add_emotion("anger", intensity)
                if rel.like < 0:
                    # 反复负面归因且好感已为负 → hate（衰减更慢, Gamygdala 语义）
                    agent.add_emotion("hate", intensity,
                                      decay_factor=float(self.config["hate_decay_factor"]))
                rel.like = clamp(rel.like - intensity * gain)
            else:
                agent.add_emotion("gratitude", intensity)
                rel.like = clamp(rel.like + intensity * gain)
            produced.append({"causal": causal, "emotion":
                             ("hate" if (total_desirability < 0 and rel.like < 0) else
                              ("anger" if total_desirability < 0 else "gratitude")),
                             "intensity": round(intensity, 3),
                             "like_after": round(rel.like, 3)})

        return {"produced": produced, "attitude": self.attitude(agent, causal)}

    # ---- 衰减：每游戏日调一次 ----
    def decay(self, agent: Agent) -> None:
        rate = float(self.config["decay_per_day"])
        for emo in list(agent.emotions.values()):
            # hate 的 decay_factor<1 → 衰得更慢（语义：恨更持久）
            emo.intensity = emo.intensity * (1.0 - rate * emo.decay_factor)
            if emo.intensity < 0.01:
                del agent.emotions[emo.name]

    # ---- 态度：量化合成 → 词表 ----
    def attitude(self, agent: Agent, other: str) -> Dict[str, Any]:
        rel = agent.relation(other) if other else Relation()
        gain = float(self.config["attitude_gain"])
        anger = agent.emotions.get("anger", Emotion("anger")).intensity
        hate = agent.emotions.get("hate", Emotion("hate")).intensity
        gratitude = agent.emotions.get("gratitude", Emotion("gratitude")).intensity
        score = clamp(rel.like - gain * (anger + hate) + gain * gratitude)
        return {"other": other, "like": round(rel.like, 3),
                "anger": round(anger, 3), "hate": round(hate, 3),
                "gratitude": round(gratitude, 3),
                "attitude_score": round(score, 3),
                "attitude": attitude_word(score)}

    # ---- 从人物数据构建 agent ----
    @staticmethod
    def agent_from_persona(npc_id: str, persona: Dict[str, Any]) -> Agent:
        goals = [Goal(name=str(g["name"]),
                      utility=float(g.get("utility", 0.5)),
                      likelihood=float(g.get("likelihood", 1.0)),
                      is_incremental=bool(g.get("is_incremental", False)))
                 for g in persona.get("goals", [])]
        return Agent(name=npc_id, faction=str(persona.get("faction", "")), goals=goals)

    @staticmethod
    def engine_from_persona(persona: Dict[str, Any]) -> "EmotionEngine":
        return EmotionEngine(persona.get("emotion_engine", {}))
