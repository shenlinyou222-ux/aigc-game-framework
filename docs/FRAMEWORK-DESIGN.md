# AIGC Demo 核心框架（v0.0.2）

目标不是直接接 AI，而是先把“游戏操作接口”和“事件向量/记忆/日志格式”固定下来。
AI 作为上层通过 `MainController` 调用，不会反向侵入游戏主循环。

## 结构

```
aigc_demo_framework/
├─ aigc_core/
│  ├─ models.py            # 事件向量、记忆、日志、查询、游戏操作等 Pydantic 模型
│  ├─ memory.py            # 公共向量知识库（先内存版，含生命周期/写入权限）
│  ├─ log_vector_store.py  # 日志向量库：Godot game_events.jsonl 的存储/召回/JSONL 持久化
│  ├─ embedders.py         # 向量化器：OllamaLogEmbedder（bge-m3 / mxbai-embed-large 等）
│  ├─ game.py              # 游戏适配接口 + MockGame
│  ├─ godot_game.py        # 真游戏适配：FileCommandGameAdapter（写 Godot inbox 命令通道）
│  ├─ mcel.py              # mini-CEL（纯 Python）：参考 CEL 的简易规则求值
│  ├─ micro_graph.py       # mini 编排：GraphSpec/navigate 绑定 + 内部函数节点 + 预检校验 + 执行 + 图展开
│  ├─ workflow.py          # 主函数式 JSON 工作流：call/map/if/assign/emit（LLM 出 JSON，执行器跑函数）
│  ├─ functions.py         # 内部函数地基：InternalFunction + Rule/CEL + RuleError(规则级报错/透传) + Registry
│  ├─ python_runner.py     # 受控执行器：AST 语法/白名单校验 + fallback（LLM 主函数 -> run() 调内部接口）
│  ├─ interface_store.py   # “接口即向量”特殊库：InterfaceSpec/InterfaceVectorStore/HashEmbedder
│  ├─ lifecycle.py         # 生命周期策略→expires_at 翻译：死期+持续时间、CRUD 策略、可热更换 Registry
│  ├─ query_normalize.py   # 06：查询轻量清洗（别名/停用词/切 2~3 raw 子句，不改语义）
│  ├─ working_set.py       # 06：Session 级 WorkingSet manifest（id/kind/chars+预算+revision）
│  ├─ fact_store.py        # 06：事实层（kind=fact 全局长期、写压缩去重）
│  ├─ planner.py           # 06：结构化 planner JSON 协议 + 历史段→WorkingSet 展开消息替换
│  └─ ctx_tools.py         # 06：注册 search_context/load_context/drop_context 口子
│  ├─ vector_db.py         # 真正的向量库适配：VectorDb 协议 + ChromaVectorDb（DB 管持久化/到期清理）
│  ├─ controller.py        # 主控：事件落库、日志落库、记忆查询、命令权限过滤
│  ├─ ai_gateway.py        # AI 上层占位接口
│  └─ api.py               # FastAPI HTTP 接口（可给 Godot/Craft/DSH 调用）
├─ tests/
│  ├─ test_framework.py
│  ├─ test_log_vector_store.py
│  ├─ test_embedders.py
│  ├─ test_context.py      # /ctx/* + 混合“准确召回”测试
│  ├─ test_lifecycle.py    # 死期+持续时间 + CRUD 策略热更换测试
│  └─ test_vector_db.py    # 向量库适配层（未装 chromadb 时验证给出清晰报错）
├─ run_demo.py
├─ demo_ai_interfaces.py   # AI 实测：接口向量召回 -> 真实 HTTP 调用 -> 调用写向量日志
├─ ingest_logs.py
└─ vector_store_lifecycle.json   # 日志向量库生命周期默认值（可直接改）
```

## 核心流程

1. 真实游戏/感知组件构造 `GameEvent` → `MainController.record_event()`
2. `MainController` 校验生命周期并把事件写入 `MemoryStore`
3. Godot 日志桥接构造 `GameLogRecord` → `MainController.record_log()`，写入日志向量库
4. AI 上层只允许通过 `MainController.game_query()` / `game_command()` 操作游戏
5. AI 如需记忆/日志，通过 `MainController.query_memory()` / `MainController.query_logs()` 读取

## 事件向量最小格式

```json
{
  "event_id": "uuid",
  "semantic_vector": [],
  "source_type": "亲眼",
  "source_faction": "player",
  "location": "farm",
  "timestamp": 1234567890.0,
  "lifecycle": "B",
  "confidence": 0.95,
  "exact_fields": {
    "type": "player_chat",
    "entity_id": "player_001"
  },
  "payload": {
    "text": "帮我把背包整理一下"
  }
}
```

`payload` 是给上层模型阅读的扩展字段；`exact_fields` 用于强过滤。

## 日志向量最小格式

`LogVectorStore` 保存的是 Godot Telemetry 原始日志 + `semantic_vector`：

```json
{
  "event_id": "player.move.67209580",
  "event_type": "player.move",
  "time": "2026-09-02T19:22:43",
  "unix_time": 1788348163.311,
  "frame": 3694,
  "data": {
    "direction": [1.0, 0.0],
    "moving": true
  },
  "semantic_vector": [],
  "last_used_at": 1788348163.311
}
```

- `semantic_vector` 为空时，只能做 event_type/时间过滤；接入 Ollama embedding 后可做余弦召回。
- `ingest_jsonl(source, embedder)` 可批量导入 Godot 日志并为每条日志补向量。
- 可用 `OllamaLogEmbedder(model="bge-m3")` 调用本机 Ollama 生成向量。
- `last_used_at` 是生命周期“空闲计时基准”（默认=写入时间），不覆盖原始 `unix_time`。

## 日志向量库生命周期管理（何时删）

只负责“向量啥时候该删”，当前单库；后续做多库路由时把 `StoreLifecycleConfig` 按库分别传即可。

规则：
- 空闲超时删除：距 `last_used_at` 超过 `retention_sec` 秒未用即删（`sweep_expired`，add/query 时也会懒删）。
- 按条数上限淘汰：超过 `max_entries` 按“最久未被使用”淘汰（`evict_over_cap`）。
- 警戒线：超过 `int(max_entries * warn_ratio)` → `warning_breached()` / `capacity_status().over_warning`。
- 被用即续命：把某条向量真正注入上下文前调用 `touch(event_id)`（主控侧 `touch_log`），刷新 `last_used_at`。

配置（默认在 `vector_store_lifecycle.json`，可直接改；也可 `LogVectorStore(..., config_path=...)` 传别的位置）：

```json
{
  "retention_sec": 1800,
  "max_entries": 20000,
  "warn_ratio": 0.8
}
```

主控提供 `MainController.cleanup_logs()` 供外部定时任务周期调用。

## 让 AI 触发真游戏（Python 主控 -> Godot）

- 主控只发命令，效果由 GDScript 执行（换音乐、建新物品等）。
- `FileCommandGameAdapter` 把 AI 命令追加写到 Godot `user://commands/inbox.jsonl`；Godot 侧 `CommandServer` autoload 轮询执行后清空。
- 命令：`play_music {track, volume_db?}`（assistant 可发）、`create_item {item_id, name, icon_path, count?}`（仅 system）、`give_item {item_id, count}`。
- 关联 Godot：`scripts/global/command_server.gd`、`scripts/global/music_manager.gd`、`data.gd` 的 `CUSTOM_ITEMS`。

```bash
# 主控挂上真游戏适配器即可
MainController(game_adapter=FileCommandGameAdapter())
```

## mini 编排内核（参考 graphrunner 精简）

内部接口即节点 + 显式 JSON 绑定 + 校验=检索+关系判断。

```python
from aigc_core import micro_graph as g
from aigc_core.mcel import evaluate

graph = g.GraphSpec(
    nodes=[g.NodeSpec("pick", "pick_ore"), g.NodeSpec("make", "make_item")],
    bindings=[
        g.Binding("START", "player.location", "pick", "loc"),
        g.Binding("pick", "Data.items[0].kind", "make", "item_kind"),
    ],
    input={"player": {"location": "mine"}},
)
handlers = {
    "pick_ore": lambda in_: {"Data": {"items": [{"kind": "copper"}]}},
    "make_item": lambda in_: {"item_id": "test_item"},
}
bound = g.preflight(graph, rules={"make": "item_kind in ['copper','iron']"})  # 先绑死+校验
res = g.run(graph, handlers)   # 同步执行；异步用 g.arun
```

- `navigate(data, "Data.items[0].kind")`：JSON 路径绑定（对应 graphrunner NavigatePath）
- `run`/`arun`：拓扑执行，哪步错抛 `NodeError`（带 step 日志，不续跑）
- `mcel.evaluate(expr, ctx)`：纯 Python 简易 CEL（size/contains/startsWith/has/in/短路…）

## 接口即向量（特殊接口向量库）

把 AI 可调用的接口 schema 向量化进专门库，供自然语言召回后再真实调用。

```python
from aigc_core import InterfaceVectorStore, HashEmbedder, default_interface_specs
store = InterfaceVectorStore(path="data_interfaces.jsonl", embedder=HashEmbedder())
store.add_many(default_interface_specs())
hits = store.query("创建一种叫 test 的新物品种类", top_k=3)  # -> create_item 排前列
```

`demo_ai_interfaces.py`：AI 实测闭环（召回→HTTP 调用→调用写日志向量），运行：
```bash
E:\py312\python.exe demo_ai_interfaces.py
```

单测覆盖（coverage）：整体 86~87%，api / ai_gateway / godot_game / models 100%。

## 内部函数地基（functions.py）

编排留在 Python；每个可调内部接口定义成带 CEL 前置规则的 `InternalFunction`，env 可每次不同，报错精确到“撞了哪条规则”，且下游 `RuleError` 原样透传。

```python
from aigc_core.functions import FunctionRegistry, InternalFunction, Rule

give = InternalFunction(
    name="give_item",
    description="发物品",
    env_fields=["actor", "catalog"],
    rules=[Rule(id="item_exists", expr="item_id in catalog"),
           Rule(id="actor_allowed", expr="actor in ['system']")],
    fn=lambda ctx: {"ok": True},
)
reg = FunctionRegistry().register(give)
reg.call("give_item", {"item_id": "test_item"}, {"actor": "system", "catalog": ["test_item"]})  # ok
# 换 env 或越界 -> RuleError(give_item, item_exists / actor_allowed, expr, context)
```

## 受控执行器（python_runner.py）

LLM 写 Python 主函数，只允许经 `run("函数名", params)` 调注册的内部函数。

```python
from aigc_core.python_runner import run_python

src = '''def main(ctx):
    items = run("query_items", {})["items"]
    return [run("make_item", {"id": it["id"], "kind": it["kind"]})
            for it in items if it["kind"] in ("copper","iron")]
'''
out = run_python(src, reg, env={"actor":"assistant"})          # 成功
out = run_python(bad_src, reg, env={}, on_error=lambda e: {"ok":False})  # fallback
```

限制：禁止 import / 属性访问(用下标) / 调非安全函数；只做“简单语法校验 + fallback”，超时/深度防护可后续按需加。

## 操作权限示例

| Actor | 允许 | 不允许 |
|---|---|---|
| assistant（小灰） | send_message / send_npc_dialogue / show_emote / organize_inventory / set_game_setting / play_music | give_item / create_item / set_npc_confidence / delete_memory / advance_mainline |
| system（主控） | 以上全部 + give_item / create_item / set_npc_confidence | 由具体游戏再约束 |

## HTTP 接口

```bash
E:\py312\python.exe -m uvicorn aigc_core.api:create_app --factory --host 127.0.0.1 --port 8000
```

可用接口：
- `POST /v1/events`
- `POST /v1/memory/query`
- `POST /v1/logs`
- `POST /v1/logs/query`
- `POST /v1/game/query`
- `POST /v1/game/command`
- `GET /v1/context?player_id=player_001`
- DSH 上下文后端：
  - `POST /ctx/ingest`：写一条上下文事件（输入/输出/调用/结果）
  - `POST /ctx/recall`：混合“准确召回”（dense+词法+时效+本地/角色加权）
  - `POST /ctx/touch`：命中并注入前续命（重置死期=now，duration 不变）
  - `POST /ctx/cleanup`：生命周期清理
  - `GET /ctx/status`：容量/警戒线
  - 向量生命周期（需 `MainController(log_lifecycle=VectorLifecycleManager(...))` 才启用）：
    - `GET /ctx/lifecycle/status`
    - `GET /ctx/lifecycle/meta/{event_id}`
    - `POST /ctx/lifecycle/duration` `{event_id, duration_sec}`
    - `POST /ctx/lifecycle/reset` `{event_id}`
  - 06 检索与工具组装：
    - `POST /ctx/query_normalize` `{text}`
    - `POST /ctx/fact/ingest` `{text, tags?}`、`POST /ctx/fact/search` `{text, limit}`
    - `POST /ctx/tool/search` `{text, top_k}`
    - `POST /ctx/working-set` `{session}`、`POST /ctx/working-set/load` `{session, ids}`
    - `POST /ctx/working-set/drop` `{session, ids}`、`POST /ctx/working-set/expand` `{session}`
    - `POST /ctx/assemble` `{session, user_text, scope?, auto_load?}`

（DSH 上下文机制见 `../aigc_docs/03-向量上下文机制与DSH插件方案.md` 与 `../aigc_docs/04-DSH全量日志与上下文准确召回方案.md`）

## 运行

```bash
E:\py312\python.exe -m unittest discover -s tests -v
E:\py312\python.exe run_demo.py
```

## 06 机制（本会话落地，见 `../aigc_docs/06-会话上下文检索与工具组装机制.md`）

职责拆分：LLM=意图+取舍；查询=`query_normalize` 轻清洗后原样进向量库（不改写）；质量=后端 recall_hybrid+分层；程序=`working_set`+`planner.assemble_messages` 负责组装；工具与事实同构（scope: fact/event/tool），`assemble_context` 一并查出；留 `search_context`/`load_context`/`drop_context` 给 LLM 主动补。单测 129 通过（含 `tests/test_06_mechanisms.py`）。

## 下一步

- 用 `OllamaLogEmbedder` 实际灌一批带向量的日志，验证召回效果
- 接入 Ollama LLM，实现 `AILayer.handle_player_input`
- 把 Godot `game_events.jsonl` 增量桥接到 `LogVectorStore`
- 把 `MockGame` 换成真实游戏适配器
