# aigc-game-framework · 让 AI 安全地操作游戏

一个用 Python 写的游戏 AI 中间层框架。核心是一个**架构约束**，而不是一堆功能：

> **AI 永远不能侵入游戏主循环。**
> 它只能通过 `MainController` 读取记忆/日志、下发指令，而每条指令都要过一次规则校验。

这个约束的具体含义、为什么值得为它付出复杂度，以及它带来了什么可测的性质，是本仓库的主要内容。

**测试状态：`223 passed`（12.5 秒，零失败）** —— 可复现，见下方「快速验证」。

---

## 为什么需要它

把 LLM 接进游戏，最省事的做法是让模型直接调游戏函数。这条路会在三个地方崩掉：

| 问题 | 直连式做法的后果 |
|---|---|
| **幻觉** | 模型编出一个不存在的物品 ID，游戏直接崩或行为诡异 |
| **越权** | 模型写了本该只读的存档字段，出了 bug 无法定位是谁改的 |
| **不可复现** | 同样的输入因为模型采样随机，产生不同的游戏状态，无法回归测试 |

本框架的解法是**把模型降级为「提议者」**，把裁决权交给确定性的规则层：

```
游戏事件 ──► MainController.record_event() ──► MemoryStore（带生命周期）
                    ▲
                    │ 只读查询 / 指令下发（都要过权限过滤）
                    │
              AI 层（模型在这里，但只能提议）
                    │
                    ▼
             受控执行器（AST 校验 + 白名单 + CEL 规则裁决）
                    │
                    ▼
             接口向量库 ──► 真实 HTTP 调用
```

规则用 **mini-CEL**（参考 Google CEL 的纯 Python 实现）表达，所以**权限与门控是数据，不是代码** —— 改规则不用改框架。

---

## 已验证的性质（都能在本仓库复现）

| 性质 | 怎么保证的 | 对应测试 |
|---|---|---|
| **只有本人能写自己的记忆** | CEL 规则门控 | `test_npc_memory.py::test_self_write_only_by_cel` |
| **过短/无意义的输入被拒绝** | 写入前置校验 | `test_short_text_rejected_by_cel` |
| **LLM 生成的 Python 不能为所欲为** | AST 语法 + 白名单校验 | `test_python_runner.py` |
| **上下文有预算，不会无限膨胀** | WorkingSet manifest + 字符预算 | `test_context.py` |
| **记忆会按策略过期** | 生命周期：死期 + 持续时间 | `test_lifecycle.py` |
| **向量库缺失时给出清晰报错** | 适配层契约 | `test_vector_db.py` |

---

## 快速验证

```bash
pip install -r requirements.txt
python -m pytest tests -q
# 预期：223 passed
```

> 复现环境：Python 3.12 / Windows。测试不依赖 GPU、不依赖本机 Ollama（embedding 走可替换的适配器，测试用 HashEmbedder）。

---

## 目录结构

```
aigc_core/                   核心库（30 个模块）
├─ models.py                 事件向量 / 记忆 / 日志 / 查询 / 指令的 Pydantic 模型
├─ controller.py             主控：事件落库、日志落库、记忆查询、指令权限过滤
├─ memory.py                 公共向量知识库（生命周期 + 写入权限）
├─ log_vector_store.py       日志向量库：存储 / 召回 / JSONL 持久化
├─ embedders.py              向量化器适配（Ollama bge-m3 等，测试可替换）
├─ vector_db.py              向量库协议 + Chroma 实现
├─ mcel.py                   mini-CEL：参考 CEL 的简易规则求值
├─ functions.py              内部函数地基：Rule/CEL + 规则级报错 + Registry
├─ python_runner.py          受控执行器：AST 校验 + 白名单 + fallback
├─ workflow.py               函数式 JSON 工作流：call/map/if/assign/emit
├─ micro_graph.py            mini 编排：GraphSpec + 绑定推导 + 预检 + 执行
├─ interface_store.py        「接口即向量」库
├─ lifecycle.py              生命周期策略 → expires_at 翻译，可热更换
├─ query_normalize.py        查询轻量清洗（别名 / 停用词 / 子句切分）
├─ working_set.py            Session 级 WorkingSet manifest（id/kind/chars + 预算）
├─ fact_store.py             事实层（全局长期，写时压缩去重）
├─ planner.py                结构化 planner JSON 协议 + 历史段 → 消息替换
├─ ctx_tools.py              search/load/drop context 三个口子
├─ game.py / godot_game.py   游戏适配接口 + Godot inbox 命令通道
├─ npc_memory.py             NPC 记忆：置信度、写权限、主动召回门控
├─ npc_emotion.py            情绪路由
├─ ai_gateway.py             AI 上层接口
└─ api.py                    FastAPI HTTP 接口

tests/                       24 个测试文件 / 223 个用例
tool_registry/               33 个工具规格（JSON，数据驱动）
packs/                       内容包定义
skills/                      Agent 技能定义
emotion_dict/                情绪词典
sticker_lora/                表情包 LoRA 数据集准备
docs/                        原始设计文档与服务说明
```

---

## 两个设计细节

### ① 事件与日志是两种东西，不能混

- **事件向量**（`GameEvent`）：语义化的、给模型读的。「玩家说了什么」「NPC 做了什么决定」
- **日志向量**（`GameLogRecord`）：原始遥测帧数据。「第 3694 帧玩家朝 x 方向移动」

分开存的原因是**召回方式不同**：事件靠语义相似度召回，日志靠 `event_type` + 时间窗过滤。混在一起会让两边都变差。

### ② 完整性与合法性来自「派生」，不来自「模型多输出」

模型只输出**最小意图**（意图 + 参数 + 目标），作业层的 `tasks` / `context vars` / `nesting` 由编排器**派生**出来。

这样做的收益：模型的输出空间变小了，幻觉的余地就小了；而完整性由确定性代码保证，可以写测试。

---

## 已知边界 / 未完成

诚实清单：

- **模型本身不在此仓库**：所有推理依赖本地服务（Ollama / vLLM），本仓库只有客户端与适配层。`models/` 与评测数据已排除（合计 7 GB）。
- **`mini-CEL` 不是完整 CEL**：实现了本项目需要的子集（比较、逻辑、成员判断），**不支持**宏、自定义函数与类型系统。别拿它当通用规则引擎。
- **Godot 侧代码不在本仓库**：`godot_game.py` 走的是文件 inbox 通道，游戏端实现见游戏 Demo 仓库。
- **部分顶层脚本是服务启动器**，依赖本机端口与模型文件，单独跑不起来（`voice_stt_server.py` / `music_gen_service.py` 等）。它们是集成示例，不是可独立运行的库。
- **测试是单元/集成级**：223 个用例覆盖了框架的规则与数据流，**没有端到端跑真实游戏**的自动化测试。
- **Windows 优先**：路径处理与进程管理按 Windows 写的，未在 Linux 上验证。

---

## 许可

MIT
