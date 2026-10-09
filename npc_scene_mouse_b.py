# -*- coding: utf-8 -*-
"""AI-NPC 召回推理实测：玩家杀老鼠A → 老鼠B 访问到事实 → 预设向量触发 → qwen 推理态度
→ 新事实向量入库 → 组装上下文生成简短对话（含触发条件）。

分工：触发=CEL（预设相关度阈值）｜推理=qwen3.5:0.8b（结论由模型产出）｜入库=CEL（权限+门槛）
用法: python npc_scene_mouse_b.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

from aigc_core.embedders import OllamaLogEmbedder
from aigc_core.interface_store import HashEmbedder
from aigc_core.llm_json import (
    DIALOGUE_SCHEMA,
    INFERENCE_SCHEMA,
    adapt_dialogue,
    adapt_inference,
    generate_json,
)
from aigc_core.log_vector_store import LogVectorStore
from aigc_core.npc_memory import NpcMemory, build_inference_prompt, build_npc_prompt

OLLAMA = "http://127.0.0.1:11434"
LLM_MODEL = "qwen3.5:0.8b"
STORE_PATH = Path(__file__).parent / "npc_memory_demo.jsonl"
REPORT_PATH = Path(__file__).parent / "npc_scene_report.json"


def log(step: str, msg: str = "") -> None:
    print(f"\n========== {step} ==========")
    if msg:
        print(msg)


def ollama_generate(prompt: str, adapter, schema):
    """qwen3.5:0.8b 常驻(keep_alive=-1) + 结构化输出 + 三层 JSON 修复适配。

    返回 (适配对象 | None, 原始输出, 审计)。失败降级链在 llm_json 内部处理。
    """
    obj, audit = generate_json(
        prompt,
        adapter=adapter,
        json_schema=schema,
        model=LLM_MODEL,
        keep_alive=-1,          # 0.8b 常驻: 防 5s 级冷加载 (2026-09-05 决策; 整数, 字符串 "-1" 会 400)
        num_ctx=2048,           # NPC prompt ≤1k token, 32k ctx 实测多吃 0.45GB 显存
    )
    repairs = audit.get("repairs", [])
    if repairs:
        print(f"[llm_json] 修复层触发: {repairs}")
    if audit.get("format_fallback"):
        print("[llm_json] 结构化输出降级: format=json 重试")
    return obj, audit.get("raw", ""), audit


def main() -> int:
    # ---------- 准备：全新演示库 + qwen3-embedding 真向量 ----------
    if STORE_PATH.exists():
        STORE_PATH.unlink()
    try:
        embedder = OllamaLogEmbedder(model="qwen3-embedding:0.6b")
        embedder.embed_text("连通性测试")
        print("[setup] embedder = qwen3-embedding:0.6b (Ollama)")
    except Exception as exc:  # Ollama 不可用则退化确定性向量
        embedder = HashEmbedder()
        print(f"[setup] qwen3-embedding:0.6b 不可用({exc}), 退化 HashEmbedder")
    personas = json.loads((Path(__file__).parent / "npc_personas.json").read_text("utf-8"))["npcs"]
    store = LogVectorStore(path=STORE_PATH)
    nm = NpcMemory(store, personas, embedder=embedder, actor="system")
    report: dict = {}

    # ---------- S1 世界事实入库：玩家干掉老鼠A ----------
    log("S1 世界事实入库（含阵营标记, 系统公告=全员可见）")
    fact_text = "玩家在农田边杀死了老鼠A"
    fact_id = nm.ingest_world_fact(
        fact_text,
        source_type="系统公告",
        faction="mouse",
        event_type="world.enemy_died",
        extra={"victim": "mouse_a", "killer": "player"},
    )
    print(f"fact event_id = {fact_id}\nfact text     = {fact_text}")
    report["S1_fact"] = {"event_id": fact_id, "text": fact_text}

    # ---------- S2 老鼠B 主动召回（两道 CEL 门） ----------
    log("S2 老鼠B 主动召回（阵营访问门 + 记忆门）")
    r1 = nm.recall("mouse_b", "玩家 老鼠 同伴")
    print(f"access_gate = {r1['access_gate']}")
    print(f"denied_by_access = {r1['denied_by_access']}  dropped_by_gate = {r1['dropped_by_gate']}")
    print(f"memory_block:\n{r1['memory_block']}")
    assert fact_id in [it["event_id"] for it in r1["items"]], "事实应过两道门进入记忆块"
    report["S2_recall"] = {"items": r1["items"], "denied": r1["denied_by_access"]}

    # ---------- S3 预设向量触发（CEL: sim >= threshold） ----------
    log("S3 预设向量相关度（触发器, 不产生结论）")
    pt = nm.preset_triggers("mouse_b", fact_text)
    for s in pt["scored"]:
        mark = "★触发" if s["sim"] >= s["threshold"] else "未达阈值"
        print(f"  preset[{s['id']}] sim={s['sim']:.3f} vs threshold={s['threshold']} -> {mark}")
    if not pt["triggered"]:
        print("未触发任何预设，流程结束（无推理）")
        return 1
    trig = pt["triggered"][0]
    print(f"触发信条: [{trig['id']}] {trig['text']} (sim={trig['sim']}, hint={trig['hint']})")
    report["S3_preset"] = pt

    # ---------- S3.5 Gamygdala 式情绪引擎：量化情绪系数 ----------
    log("S3.5 情绪引擎（Gamygdala 式评价）: 目标效用×事件增量×概率 → 情绪/关系/态度系数")
    from aigc_core.npc_emotion import EmotionEngine, Agent, Goal
    persona_mb = personas["mouse_b"]
    agent_mb = EmotionEngine.agent_from_persona("mouse_b", persona_mb)
    engine = EmotionEngine.engine_from_persona(persona_mb)
    appraisal = engine.appraise(
        agent_mb,
        {"causal": "player", "target": "mouse_a", "likelihood": 1.0,
         "goals": {"同族安全": -0.8, "自身生存": -0.2}},
        "mouse_b",
    )
    att_readout = appraisal["attitude"]
    print(f"引擎产出: {appraisal['produced']}")
    print(f"态度读数: like={att_readout['like']} anger={att_readout['anger']} "
          f"hate={att_readout['hate']} → score={att_readout['attitude_score']} "
          f"({att_readout['attitude']})")
    report["S3_5_engine"] = appraisal

    # ---------- S4 大模型推理（qwen3.5:0.8b）：态度结论由模型产出 ----------
    log(f"S4 提交 qwen 推理（{LLM_MODEL}）：信条+事实 → 态度结论")
    old_attitude = "中性"
    inf_prompt = build_inference_prompt(personas["mouse_b"], trig, fact_text, old_attitude)
    inf_prompt += (
        "\n情绪引擎读数（必须与读数一致）：你对玩家的愤怒强度 "
        f"{att_readout['anger']:.2f}，恨 {att_readout['hate']:.2f}，"
        f"关系值 {att_readout['like']:+.2f}（引擎态度倾向：{att_readout['attitude']}）。"
    )
    print("--- 推理 prompt ---")
    print(inf_prompt)
    verdict, raw, inf_gen_audit = ollama_generate(inf_prompt, adapt_inference, INFERENCE_SCHEMA)
    if verdict is None:
        raise RuntimeError(f"推理输出经三层修复仍不可解析: {raw[:200]}")
    print("--- 模型原始输出 ---")
    print(raw)
    attitude = verdict.attitude
    new_memory_text = verdict.new_memory
    model_conf = verdict.confidence
    if model_conf is not None:
        conf_note = "模型自报"
    else:
        # 小模型常漏报: 落人物数据默认推理置信(门槛照卡, 不放水)
        model_conf = float(personas["mouse_b"]["inference"]["base_confidence"])
        conf_note = f"模型未自报 → 用人物数据默认 {model_conf}"
    print(f"模型结论: attitude={attitude}  confidence={model_conf} ({conf_note})")
    print(f"模型新事实: {new_memory_text}")
    report["S4_inference"] = {"prompt": inf_prompt, "raw": raw,
                              "verdict": verdict.model_dump(), "gen_audit": inf_gen_audit}

    # ---------- S5 推理结论过 CEL 门槛入库（新事实向量） ----------
    log("S5 推理结论过 CEL 门槛入库（self_write / grounded / 置信）")
    st = nm.store_inference(
        "mouse_b",
        text=new_memory_text,          # ← 模型的原话, 程序不代写
        grounded=True,                 # 证据事实已在库（正式版由 claim_checker 判定）
        confidence=model_conf,         # 模型自报, 门槛再卡一道
        attitude=attitude,
        model=LLM_MODEL,
        evidence=[fact_id],
        actor="mouse_b",               # NPC 自己写（self_write_only 放行本人）
        extra_data={"emotion_engine": att_readout},  # 情绪系数随事实入库(可审计)
    )
    print(f"stored = {st['stored']}  event_id = {st.get('event_id')}")
    print(f"session = {st.get('session')}  confidence = {st.get('confidence')}  gate = {st.get('gate')}")
    assert st["stored"], "模型结论应过门槛入库"
    report["S5_store"] = st

    # ---------- S6 复查：新事实已可被召回 ----------
    log("S6 复查召回（新事实向量已可检索）")
    r2 = nm.recall("mouse_b", f"老鼠B 对玩家 {attitude} {new_memory_text}")
    print(f"memory_block:\n{r2['memory_block']}")
    ids = [it["event_id"] for it in r2["items"]]
    assert st["event_id"] in ids, "推理新事实必须能被再次召回（入库成功且语义可达）"
    kept_item = next(it for it in r2["items"] if it["event_id"] == st["event_id"])
    print(f"✓ 推理新事实可召回: [{kept_item['source_type']}|置信{kept_item['confidence']:.2f}] {kept_item['text']}")
    report["S6_recheck"] = {"items": r2["items"]}

    # ---------- S7 组装上下文 → qwen 生成简短对话（含触发条件） ----------
    log(f"S7 组装上下文提交 {LLM_MODEL} 生成对话")
    # 上下文 = 角色卡 + 预设触发语境 + 最新记忆块（含仇恨新事实）
    memory_block = r2["memory_block"] + f"\n- [信条触发|sim{trig['sim']}] 你的信条「{trig['text']}」被触碰 ({trig['hint']})"
    dlg_prompt = build_npc_prompt(personas["mouse_b"], memory_block, "（玩家朝你走近了两步）")
    print("--- 对话 prompt ---")
    print(dlg_prompt)
    dlg, raw2, dlg_gen_audit = ollama_generate(dlg_prompt, adapt_dialogue, DIALOGUE_SCHEMA)
    if dlg is None:
        raise RuntimeError(f"对话输出经三层修复仍不可解析: {raw2[:200]}")
    print("--- 模型原始输出 ---")
    print(raw2)
    lines = list(dlg.lines)
    emotion_raw = dlg.emotion
    trigger = dlg.trigger or {"type": "proximity", "radius_px": 64}
    dlg_claims = dlg.claims
    # 情绪归一(抄现成的 chat_emotion_router 质心路由): 模型自由文本 -> 闭集13类贴纸标签
    emo_tag, emo_score = "", -1.0
    try:
        import chat_emotion_router as cer
        emo_tag, emo_score = cer.route_emotion(" ".join(lines) + " " + emotion_raw)
    except Exception as exc:
        print(f"[emotion归一] 质心路由不可用: {exc}")
    print(f"对话: {lines}\n表情(模型自由文本): {emotion_raw}")
    print(f"表情(归一后闭集标签): {emo_tag} (score={emo_score})")
    print(f"触发条件: {trigger}\nclaims: {dlg_claims}")
    report["S7_dialogue"] = {"prompt": dlg_prompt, "raw": raw2, "dialogue": dlg.model_dump(),
                             "gen_audit": dlg_gen_audit,
                             "emotion_tag": emo_tag, "emotion_score": emo_score}

    # ---------- S8 产出游戏侧命令（set_dialog + 触发条件元数据） ----------
    log("S8 产出游戏 inbox 命令（set_dialog + 触发条件）")
    command = {
        "command_id": f"npc_hatred_{int(__import__('time').time())}",
        "command_type": "set_dialog",
        "actor": "system",
        "params": {
            "npc": personas["mouse_b"]["node"],       # 游戏 NPC 节点名
            "lines": lines,
            "reset_index": True,
            # 触发条件元数据(供 AI 层/CommandServer 扩展使用):
            "trigger": trigger,
            "emotion": emotion_raw,
            "emotion_tag": emo_tag,                   # 归一后的闭集贴纸标签
            "emotion_score": emo_score,
            "attitude": attitude,
            "source_fact": fact_id,
            "inference_event": st.get("event_id"),
        },
    }
    cmd_json = json.dumps(command, ensure_ascii=False)
    print(cmd_json)
    report["S8_command"] = command

    REPORT_PATH.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    log("DONE", f"报告已写 {REPORT_PATH.name}；演示库 {STORE_PATH.name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
