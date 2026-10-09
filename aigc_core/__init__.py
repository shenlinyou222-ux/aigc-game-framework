"""AIGC 游戏 Demo 核心框架。

该框架只负责主控、事件向量、游戏操作接口、记忆存储和日志向量库；
AI 作为上层能力通过 MainController 暴露的接口接入，不侵入游戏主循环。
"""

from .controller import MainController
from .models import (
    ActionResult,
    ActorRole,
    ExactFields,
    GameCommand,
    GameEvent,
    GameLogRecord,
    GameQuery,
    Lifecycle,
    LogSearchResult,
    LogVectorQuery,
    MemoryEntry,
    MemoryQuery,
    SourceType,
    WriterRole,
)
from .embedders import OllamaLogEmbedder
from .functions import (
    FunctionRegistry,
    InternalFunction,
    Rule,
    RuleError,
    call_internal,
)
from .game import GameAdapter, MockGame
from .godot_game import DEFAULT_INBOX_PATH, FileCommandGameAdapter
from .interface_store import (
    HashEmbedder,
    InterfaceHit,
    InterfaceSpec,
    InterfaceVectorStore,
    default_interface_specs,
)
from .log_vector_store import (
    DEFAULT_LIFECYCLE_CONFIG_PATH,
    LogVectorStore,
    StoreLifecycleConfig,
    load_lifecycle_config,
    write_lifecycle_config,
)
from .fact_store import FactStore
from .planner import (
    PlanResult,
    assemble_messages,
    build_planner_input,
    parse_plan,
    split_id,
)
from .ctx_tools import register_context_tools
from .query_normalize import QueryNormalizer, query_normalize
from .working_set import WorkingSetManifest, WorkingSetService
from .memory import InMemoryMemoryStore
from .python_runner import PythonRunnerError, run_python

__all__ = [
    "ActionResult",
    "ActorRole",
    "DEFAULT_INBOX_PATH",
    "DEFAULT_LIFECYCLE_CONFIG_PATH",
    "ExactFields",
    "FileCommandGameAdapter",
    "FunctionRegistry",
    "GameAdapter",
    "GameCommand",
    "GameEvent",
    "GameLogRecord",
    "GameQuery",
    "HashEmbedder",
    "InMemoryMemoryStore",
    "InterfaceHit",
    "InterfaceSpec",
    "InterfaceVectorStore",
    "InternalFunction",
    "Lifecycle",
    "Rule",
    "RuleError",
    "LogSearchResult",
    "LogVectorQuery",
    "LogVectorStore",
    "MainController",
    "MemoryEntry",
    "MemoryQuery",
    "MockGame",
    "OllamaLogEmbedder",
    "PythonRunnerError",
    "SourceType",
    "StoreLifecycleConfig",
    "WriterRole",
    "call_internal",
    "default_interface_specs",
    "load_lifecycle_config",
    "run_python",
    "FactStore",
    "PlanResult",
    "QueryNormalizer",
    "WorkingSetManifest",
    "WorkingSetService",
    "assemble_messages",
    "build_planner_input",
    "parse_plan",
    "query_normalize",
    "register_context_tools",
    "split_id",
    "write_lifecycle_config",
]
