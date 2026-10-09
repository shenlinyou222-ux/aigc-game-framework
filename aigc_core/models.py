"""核心数据结构定义。

这里先固定“事件向量”和“游戏操作接口”的 JSON 形状。
AI 层、客户端、真实游戏适配器都基于这些模型通信，避免各自随意扩展。
"""

from __future__ import annotations

import time
import uuid
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class SourceType(str, Enum):
    """信源类型。"""

    EYEWITNESS = "亲眼"
    HEARD = "亲耳所听"
    SAME_FACTION_HEARSAY = "同阵营转述"
    ENEMY_FACTION_HEARSAY = "敌对阵营转述"
    SYSTEM = "系统"


class Lifecycle(str, Enum):
    """记忆生命周期。

    S: 永久，仅程序册写
    A: 长期，程序写 + 模型评分
    B: 中期，模型可写
    C: 短期，模型可写
    """

    S = "S"
    A = "A"
    B = "B"
    C = "C"


class WriterRole(str, Enum):
    """写入者角色，用于权限判定。"""

    PROGRAM = "program"        # 程序 / 主控
    MODEL = "model"            # 模型
    SENSOR = "sensor"          # 游戏内感知组件


class ActorRole(str, Enum):
    """调用方角色。AI 上层和游戏客户端都通过该角色过权限。"""

    GAME = "game"              # 游戏客户端/真实世界
    SYSTEM = "system"          # 主控
    ASSISTANT = "assistant"    # 小灰助手（AI 上层）


class ExactFields(BaseModel):
    """精确过滤字段，用于“不靠向量相似度也能查”的强过滤。"""

    type: str = Field(default="", description="事件类型/物体类型，例如 npc_dialogue、item")
    entity_id: str = Field(default="", description="实体 ID，例如 npc_777")


class MemoryEntry(BaseModel):
    """公共向量知识库中的一条结构化记忆/事实。"""

    event_id: str = Field(default_factory=lambda: uuid.uuid4().hex)
    semantic_vector: List[float] = Field(
        default_factory=list,
        description="文本/图片/状态的语义向量；未接入 embedding 时可为空",
    )
    source_type: SourceType = SourceType.SYSTEM
    source_faction: str = Field(default="", description="来源阵营/系统域")
    location: str = Field(default="", description="区域 ID，例如 village、mine")
    timestamp: float = Field(default_factory=time.time)
    lifecycle: Lifecycle = Lifecycle.C
    confidence: float = Field(default=0.9, ge=0.0, le=1.0)
    exact_fields: ExactFields = Field(default_factory=ExactFields)
    payload: Dict[str, Any] = Field(
        default_factory=dict,
        description="供上层模型阅读的原始事实/文本，不参与向量主键比较",
    )


class GameEvent(BaseModel):
    """游戏内任意感知/行为/系统事件。

    真实游戏只需把事件翻译成这个结构后调用 MainController.record_event。
    semantic_vector 可以先留空，由主控的向量化服务异步补全。
    """

    event_id: str = Field(default_factory=lambda: uuid.uuid4().hex)
    event_type: str = Field(..., description="如 player_chat、block_changed、npc_interact")
    source_type: SourceType = SourceType.SYSTEM
    source_faction: str = Field(default="player")
    location: str = Field(default="")
    entity_id: str = Field(default="")
    timestamp: float = Field(default_factory=time.time)
    lifecycle: Lifecycle = Lifecycle.B
    confidence: float = Field(default=0.9, ge=0.0, le=1.0)
    payload: Dict[str, Any] = Field(default_factory=dict)
    semantic_vector: List[float] = Field(default_factory=list)

    def to_memory_entry(self) -> MemoryEntry:
        return MemoryEntry(
            event_id=self.event_id,
            semantic_vector=self.semantic_vector,
            source_type=self.source_type,
            source_faction=self.source_faction,
            location=self.location,
            timestamp=self.timestamp,
            lifecycle=self.lifecycle,
            confidence=self.confidence,
            exact_fields=ExactFields(type=self.event_type, entity_id=self.entity_id),
            payload=self.payload,
        )


class MemoryQuery(BaseModel):
    """查询公共知识库的强过滤条件。

    向量相似度召回不在本文件实现，先提供无向量过滤接口。
    """

    event_type: Optional[str] = None
    entity_id: Optional[str] = None
    location: Optional[str] = None
    source_faction: Optional[str] = None
    lifecycle: Optional[Lifecycle] = None
    min_confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    since: Optional[float] = None
    until: Optional[float] = None
    limit: int = Field(default=20, ge=1, le=100)


class GameLogRecord(BaseModel):
    """Godot Telemetry 写入的一行结构化日志。

    形状对齐 `game_events.jsonl` 的当前字段；`semantic_vector` 由 Python
    向量化服务（如 Ollama embedding）填充分布式向量，未接入时可为空。
    DSH 上下文事件也复用本结构：event_type=事件类型/角色、kind=细分类型、
    data=原始内容、session_id=会话 ID（空=全局长期记忆）。
    """

    event_id: str = Field(..., description="事件 ID，例如 player.move.67209580")
    event_type: str = Field(..., description="事件类型，例如 player.move / npc.dialogue_line")
    time: str = Field(default="", description="Godot 可读时间字符串")
    unix_time: float = Field(default=0.0, description="Unix 时间戳")
    frame: int = Field(default=0, description="Godot 帧号")
    data: Dict[str, Any] = Field(
        default_factory=dict,
        description="原始结构化上下文，例如方向/位置/动作/对话文本",
    )
    semantic_vector: List[float] = Field(
        default_factory=list,
        description="该日志的语义向量；未接入 embedding 时为空数组",
    )
    last_used_at: float = Field(
        default=0.0,
        description=(
            "生命周期计时基准（空闲计时用）：该记录最后一次被注入上下文/被使用的时间。"
            "<=0 表示尚未使用，写入时会初始化为当前时间；不被覆盖原始 unix_time。"
        ),
    )
    session_id: str = Field(
        default="",
        description=(
            "会话 ID（DSH session / 游戏局）；空串=全局长期记忆，可被任意 session 召回。"
            "与 Godot 日志互不冲突：Godot 场景可留空，DSH 上下文场景填 session id。"
        ),
    )
    kind: str = Field(
        default="",
        description=(
            "DSH 上下文事件的细分类型（可选，便于过滤/加权），例如 tool/call、tool/result、"
            "user/message、assistant/message；为空时沿用 event_type 语义。"
        ),
    )
    tags: List[str] = Field(
        default_factory=list,
        description=(
            "受控词表标签（精确匹配过滤用），建议结构化前缀约定：npc:Vendor / map:level / "
            "lifecycle:snapshot / day:3。与语义模糊召回正交：tags 走元数据门控，"
            "语义走 dense+词法融合。"
        ),
    )
    level: str = Field(
        default="INFO",
        description=(
            "日志级别，决定是否写向量库：DEBUG=只留原始遥测文件、ingest 时跳过；"
            "INFO=正常入库；MILESTONE=入库且长保留（未来 milestone 库路由依据）。"
        ),
    )
    schema_v: str = Field(
        default="",
        description="遥测 schema 版本（Godot 端写入，用于跨版本 ingest 迁移判别）。",
    )

    def to_embedding_text(self) -> str:
        """生成用于 embedding 的纯文本（tags 非空时拼入，让语义召回可跨词表命中）。"""
        import json

        payload: Dict[str, Any] = {
            "event_type": self.event_type,
            "kind": self.kind,
            "data": self.data,
        }
        if self.tags:
            payload["tags"] = sorted(self.tags)
        return json.dumps(payload, ensure_ascii=False, sort_keys=True)


class LogVectorQuery(BaseModel):
    """日志向量库查询。

    query_vector 非空时做余弦召回；为空时只按 event_type / 时间窗口过滤，
    最近日志优先返回。
    """

    query_vector: List[float] = Field(default_factory=list)
    text: str = Field(
        default="",
        description="自然语言查询文本（用于词法补充/无向量时的召回）",
    )
    event_type: Optional[str] = None
    session_id: Optional[str] = Field(
        default=None,
        description=(
            "会话过滤：None=不过滤(可含全局+任意 session)；非空=仅该 session 与全局记录"
            "(session_id 为空串的全局长期记忆)。"
        ),
    )
    since: Optional[float] = None
    until: Optional[float] = None
    min_score: float = Field(default=0.0, ge=-1.0, le=1.0)
    limit: int = Field(default=20, ge=1, le=500)
    tags: Optional[List[str]] = Field(
        default=None,
        description=(
            "标签精确过滤（元数据门控，先于向量/词法）：None=不过滤；"
            "[] 视为不过滤；非空按 tag_match 语义匹配。"
        ),
    )
    tag_match: str = Field(
        default="all",
        description="标签匹配语义：all=全部命中(AND)；any=任一命中(OR)。标签零命中时自动降级为无标签查询。",
    )


class LogSearchResult(BaseModel):
    """日志向量召回结果。"""

    record: GameLogRecord
    score: float = Field(default=0.0, description="余弦相似度；非向量查询为 0")
    tag_matched: bool = Field(
        default=True,
        description="是否在标签门控内命中；标签零命中触发降级重查时为 False。",
    )


class HybridRecallResult(BaseModel):
    """混合召回（dense+词法+时效+实体）的结果，带各子分便于调参与审计。"""

    record: GameLogRecord
    score: float = Field(default=0.0, description="融合后的最终分")
    dense_score: float = Field(default=0.0, description="余弦相似度（未启用/缺失=0）")
    lexical_score: float = Field(default=0.0, description="词法相似度（BM25-lite 归一化）")
    recency_score: float = Field(default=0.0, description="时效分 0~1（越新越高）")
    entity_score: float = Field(
        default=0.0,
        description="实体子分（命中查询实体的归一化分数；未启用实体层=0）",
    )
    local: bool = Field(default=False, description="是否与查询同 session（本地上下文）")
    matched_terms: List[str] = Field(default_factory=list, description="命中的查询词条")
    entity_hits: List[str] = Field(
        default_factory=list,
        description="命中的实体文本（实体层加成的来源，便于审计）",
    )
    tag_matched: bool = Field(
        default=True,
        description="是否在标签门控内命中；标签零命中触发降级重查时为 False。",
    )


class GameQuery(BaseModel):
    """游戏只读查询。AI/主控不直接碰游戏对象，只发这种查询。"""

    query_id: str = Field(default_factory=lambda: uuid.uuid4().hex)
    query_type: str = Field(
        ...,
        description="get_world_state / get_player_state / get_inventory / get_npc_state / get_game_setting / list_entities",
    )
    params: Dict[str, Any] = Field(default_factory=dict)


class GameCommand(BaseModel):
    """游戏写操作/命令。AI 只能发主控允许的命令。"""

    command_id: str = Field(default_factory=lambda: uuid.uuid4().hex)
    command_type: str = Field(
        ...,
        description="send_message / send_npc_dialogue / organize_inventory / set_game_setting / give_item / ...",
    )
    params: Dict[str, Any] = Field(default_factory=dict)


class ActionResult(BaseModel):
    """查询/命令统一返回结构。"""

    ok: bool = True
    code: str = Field(default="ok", description="ok / denied / not_found / invalid / error")
    message: str = Field(default="")
    data: Dict[str, Any] = Field(default_factory=dict)
