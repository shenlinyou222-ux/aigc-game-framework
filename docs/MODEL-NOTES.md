# 模型接入备忘

## Needle 2（2026-09-06 已卸载 ❌）
- cactus_needle 2.0.11 已从 Python 3.14 卸载，~/.cache/cactus-needle DLL 缓存已删（用户拍板，不再作为 tool-calling 候选底座）。
- 原实测结论（CPU-only、必须给 say 工具、中文 max_new_tokens≥512、进程内单例）随模型一并作废。

## qwen3.5:0.8b（已接入, 见 hot_dialog.py）
- 是 Ollama 服务模型，走 `http://127.0.0.1:11434/api/chat`。
- 是 thinking 模型：请求须带 `"think": false`，否则 content 为空。
- 中文必须用 Python urllib UTF-8 直连（Git Bash curl 传参会编码损坏）。
- 0.8b 多行输出不稳，逐句单次调用 + few-shot/高温采样。
- 热修改链路：`hot_dialog.py` -> inbox.jsonl `set_dialog` -> Godot `CommandServer` 热替换 NPC 台词（无需重启游戏）。

## qwen3.5:0.8b —— NPC 推理/对话微调计划（2026-09-04 记）
- 现状：`npc_scene_mouse_b.py` 实测直接 prompting 可用但暴露三类不稳定——漏报 confidence（已用人物数据默认值兜底）、new_memory 句式不合规、格式偶崩。
- 待办①：建专项测试集 `npc_model_eval/`（推理+对话+格式鲁棒三卷，含"无关事件防过度触发"与"诱导编造"用例），自动化判卷：JSON 解析率/字段齐全/句式合规/幻觉率。
- 待办②：不达标项训 LoRA（rank 8~16，0.8B 在 8GB 显存可训；单独占卡，吸取 Anima 中断教训）；训练集=同分布自举+修正样本；产物转 GGUF adapter 挂回 Ollama。
- 达标线与用例维度详见 `../aigc_docs/12` §11；实测记录见 `../aigc_docs/12` §10。

## qwen3.5:0.8b —— 双 LoRA 蒸馏集落地（2026-09-05 记）
- `npc_model_eval/` 已建成：推理/对话**分卷**蒸馏集（messages JSONL，与 `pixel_sprite_poc/train_lora.py` 即插即用）+ 12§11 判卷器 `judge.py` + 基线评测 `run_small_model_eval.py`。
- 教师数据（会话内大模型直接产出并审查）：推理 75 train/16 eval（同族被害/被喂食/被偷窃/被救治/防过度触发，含情绪引擎读数变体），对话 59 train/21 eval（四 persona × 靠近/被攻击/送礼/闲聊/诱导编造/格式干扰）。输入一律经 `build_inference_prompt`/`build_npc_prompt` 渲染，与游戏管线逐字节对齐；emotion 训入 13 类闭集（省掉 S8 质心路由兜底）。
- **基线实测（eval 卷）**：推理 schema 6.2%/句式 0%（无关事件全部乱改态度）；对话 schema 0%/幻觉率 75%（4/4 诱导编造全中招）→ **两卷均判"训 LoRA"**（`data/baseline_summary.md`）。
- 教师纠正 11 条（S4/S7 场景实测坏输出 + 基线失败样本）已并入 train，坏输出留档 `data/corrections.jsonl` 不入训练。
- **扩量至 ~500（2026-09-05）**：新增 `teacher_gen.py` 教师条款库（事实库×句式模板×态度映射，种子 20260905 可复现）→ 推理 train 292 / 对话 train 252（合计 544）；eval 卷保持手写不变，基线报告可比。生成案例 claims 纪律：只登记台词实际陈述且可溯源记忆块的事实。
- 下一步：清显存单独占卡 → 按真 LoRA（`--r 16`）→ 训后重跑 eval 卷验收达标线（JSON≥98%/字段≥95%/幻觉≤2%/句式≥90%）。
- **训练实测（2026-09-05，复用 pixel_sprite_poc/train_lora.py，base=Qwen3.5-0.8B）**：共 6 个 adapter，
  对照表见 `npc_model_eval/data/VERSIONS.md`。当前最佳：推理 `out/lora_inference_v2`（schema 81%，
  JSON/字段/幻觉全过，句式 50% 未达）；对话最佳 `out/lora_dialogue_v4`（口吻有起色，JSON/情绪闭集/claims 仍未达标）。
  实测入口：`npc_model_eval/test_trained_model.py`（--ask 单条实测 / --eval 整卷验收）。
  教训：防过度触发×3+6ep 过拟合回退（v3）；对话 8ep 出复读环（v2）——扩数据多样性优于过采样加训。

## 语音意图识别 —— Qwen3.5:0.8B LoRA 待办（2026-09-06 记）
- **背景**：纯正则意图在前缀重叠场景互相打架（实锤：`切个南瓜` 被 `切个(歌)` 切歌规则截胡成 play_music；`切一` 同时命中切歌`切一首`与切水果）。**决策：正则只保留绝对零冲突的前缀意图**（切出/切开/切片、傳送/下雨/換一首…），重叠口型与其余意图交给 0.8B 意图 LoRA。
- **已就位基座（本期完成）**：
  - 单意图三段式框架：门控 → 前置参数搜索（读 `state/world_state.json`，游戏端 CommandServer 每 2s 导出背包/NPC/位置）→ 动作命令；搜索失败 → 统一 fallback（疑惑表情 + `find_item` 播报"背包里没有X"）。标杆实现：`cut_item`（"切出苹果"：搜背包→挥剑→消耗1产出"苹果片"x2 自定义物品）。
  - 命令 schema：`{command_id, command_type, params}` → inbox.jsonl → CommandServer 顺序执行 + Telemetry 回读。
- **待办①：构造意图微调集**。数据源：`voice_pairs.jsonl`（转写文本→最终动作，持续积累中）。必须覆盖：冲突对（切个歌/切个南瓜、切一首/切一个苹果）、繁体与 whisper 误听变体（一速=移速、調=调）、负面样本（闲聊不打扰、"无关事件防过度触发"）。格式对齐 npc_model_eval 的 messages JSONL：prompt=归一后转写，completion=单条命令 JSON 或 `{"intent":"chitchat"}`。
- **待办②：LoRA 训练**。rank 8~16（0.8B 8GB 显存可训，单独占卡）；管线复用 `pixel_sprite_poc/train_lora.py`；判卷复用 `npc_model_eval/judge.py`，新增维度：意图准确率/槽位字段齐全/命令 schema 合法率/闲聊误触发率。产物转 GGUF adapter 挂回 Ollama。
- **待办③：集成 `voice_intent.py`**：规则未命中且疑似指令（探针已有）→ Ollama chat（`think:false`，urllib UTF-8 直连，参照 `hot_dialog.py`）→ JSON 命令校验（command_type 白名单 + 参数 schema）→ inbox；失败/低置信回闲聊兜底。正则层保留为快路径。
- **明确不做**：多意图 DAG/规划器（已否决）；数字直调倍率（已否决）。

## ASR 后端替换：faster-whisper → SenseVoiceSmall（2026-09-06 完成）
- **动机**：whisper `zh` 不分陆台，短音频频繁出繁体（實錘：傳送/幫我/調/蘋），靠 TRAD2SIMP 补丁追着修；换 **SenseVoiceSmall**（funasr，中文原生简体+自带标点/情绪标签）从源头消除。
- **接入**：`voice_intent.py` 增加 `ASR_BACKEND="sensevoice"` 开关 + `_get_sv()/_asr_sensevoice()`；whisper 全路径保留为自动回退（sensevoice 异常时 `回退whisper`）。标签 `<|zh|><|Speech|>` 等用正则剥除；`use_itn=True` 自带标点（意图门控全是子串匹配，句号无影响）。
- **安装坑**：funasr 装了 `--no-deps`（防其依赖树动现有 torch），运行时依赖手动点名补齐：librosa soundfile modelscope torch_complex einops omegaconf kaldiio + hydra-core/jaconv/jamo/jieba/oss2/rapidfuzz/safetensors/sentencepiece/tensorboardX/tiktoken/transformers/umap_learn/websockets 等。numpy 2.5.2 与 funasr 声明的 <2 冲突但实测可跑，暂不钉版本。
- **A/B 实测**（同一批 wav）：`幫我切個西瓜`→`帮我切个西瓜。`；`傳送`→`传送。`→ tp 正常触发。模型首载 ~18-23s（服务冷启动），稳态转写 0.1-1s（CPU rtf 0.163，GPU 更快）。
- **遗留**：`切个/切一` 冲突口型仍降级闲聊（见上方 LoRA 待办①）；静音音频 SenseVoice 会输出"嗯。"走闲聊（whisper 时代走疑惑，行为差异可接受）。

## 动态批处理与模型常驻管理 —— model_batch_manager.py（2026-09-07 完成）
- **动机**：llm_json 旧决策 `keep_alive=-1` 让 qwen3.5 "Forever" 常驻（实测 ollama ps UNTIL=2318 年，
  占 900MB），每次特化训练前都得手动腾卡；embedding 又在语音链路上，卸早了冷载 3.5s 卡顿。
- **策略（唯一入口 `policy_keep_alive()`）**：qwen3.5:0.8b **非常驻**——每请求带 `keep_alive="45s"`
  空闲窗，批内保暖（热请求 0.44s），批后空闲自动卸载；`MBM_STRICT=1` 切 0 秒即用即卸。
  qwen3-embedding:0.6b **常驻位 30m**，被挤出后调用现场重载（实时加载，冷载实测 3.5s）。
- **动态批处理**：embed 多条**去重合并**一次 `/api/embed`（超 32 条自动分片；重复 query 高频出现，
  实测 9 入 2 算）；零散单条走 `EmbedMicroBatcher` 攒批窗（120ms）合并；chat 并发由 Ollama 单模型
  队列消化，`chat_many` 供逐句台词一次冷载跑整批。cold/warm 计数在 `STATS`。
- **训练闸门**：`pretrain` 封闸→等在途清零→卸载全部→**轮询校验**（服务端卸载异步，ps 先显示
  Stopping...）；`posttrain` 预热 embedding 常驻位并放行。封闸期间 chat/embed 抛 TrainGateClosed。
- **接线**：`llm_json.generate_json`（keep_alive=None→策略，test_llm_json 断言同步反转）、
  `hot_dialog.chat`、`embedders` 两处 /api/embed 全部改走策略。
- **真机实测（8GB 卡）**：chat 冷 3.45s/热 0.44s；embed 冷+32 条 3.54s、热+去重 0.28s；
  pretrain 后 ollama ps 空、VRAM 释放 ~900MB；45s 空闲后 qwen3.5 自动消失、embedding 保持。
- **注意**：embedding 常驻位实测占 2.2GB VRAM（含 4096 ctx 缓冲）；CLI `status/unload/load/
  pretrain/posttrain/bench`；单测 `tests/test_batch_manager.py` 8 例（桩服务器全离线），
  全库 181 例通过。

## NPC 触发链实测与桥接改造（2026-09-07）
- **链路状态**：桥接 `godot_valley_demo/tools/cel_dialog_bridge.py`（事件→CEL→召回→LLM 台词→inbox）
  与框架 S 链 `npc_scene_mouse_b.py`（权限事实→qwen3-embedding 相似度召回→推理→对话）均端到端
  跑通，但都是手动挂起，非常驻服务；真 embedding 召回此前只在框架侧。
- **桥接三处改造（本日）**：① LLM（qwen3:4b-instruct）接 `policy_keep_alive`（45s 空闲窗，
  实测 ollama ps UNTIL 从缺省 5min 变 44s 倒计时）；② 召回向量弃 mock——`event_vector.py` 新增
  `embed_real_many` 批量 /api/embed（旧单条 /api/embeddings+10s 超时在 embedding 冷载时静默退化
  mock_hash_v1，就是旧库存量 40 条 mock 的根因），入库 kind=qwen3_emb_0.6b/1024 维；③ 向量池
  改增量缓存（原每事件全量重建 O(N²)，现每 tick 只对未见事件批量嵌入一次）。
- **触发规模（真实会话 9/7 00:15–00:22，7.2min）**：事件 673 条（峰值 242/min）→ CEL 候选
  （要求 nearby_npc）仅 7 条 ≈ 1/min → 门控（Hawkes≥1.3 + 单NPC 30min + 日 5 次 + 全村 10min 1 条）
  → LLM 台词上限 6 次/小时，实测一场 2~4 次。
- **LLM 调用上限（实测吞吐推导）**：单触发串行成本 = embedding 召回 + 推理 + 台词 ≈
  **1.7s（全热）/ 12.5s（全冷，三模型各冷载一次）**；Ollama 单实例 GPU 串行 →
  **硬上限 ≈ 4.8~35 次触发/分钟（288~2100 次/小时）**，由模型吞吐决定；I/O 不构成上限
  （事件文件追加 242 条/min 是 KB 级；watch 全量 parse 673 条仅 2.8ms/tick，增量后更低）。
  45s 非常驻窗的含义：稳态触发间隔 >40s 时每次全冷 → 可持续 ~4.8/min；间隔短则转热上探 35/min。
  当前门控 6 次/h 只占吞吐上限的 0.3~2%，**不会阻塞**；游戏帧侧全程 inbox.jsonl 异步，零阻塞。
