# -*- coding: utf-8 -*-
"""mock: 从历史日志向量库捞玩家事件 -> CEL 门控召回 -> 组装推理 prompt -> qwen 生成推理结论
只读演示: 不写回 log_vector_store。用法: python mock_npc_inference.py [npc_id=cat] [query=...]
"""
import json, re, sys, time, urllib.request

from aigc_core.log_vector_store import LogVectorStore
from aigc_core.npc_memory import NpcMemory, build_inference_prompt

OLLAMA = "http://localhost:11434"

def ollama_chat(model, prompt, num_predict=400, temperature=0.4):
    body = json.dumps({"model": model, "messages": [{"role": "user", "content": prompt}],
                       "stream": False, "think": False,
                       "options": {"num_predict": num_predict, "temperature": temperature}}).encode()
    req = urllib.request.Request(OLLAMA + "/api/chat", data=body, headers={"Content-Type": "application/json"})
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=300) as r:
        d = json.loads(r.read())
    return d["message"]["content"].strip(), time.time() - t0

class OllamaBgeM3:
    """bge-m3 (1024维), 与 log_vector_store.jsonl 里 semantic_vector 同源"""
    def embed_text(self, text):
        body = json.dumps({"model": "qwen3-embedding:0.6b", "prompt": text}).encode()
        req = urllib.request.Request(OLLAMA + "/api/embeddings", data=body,
                                     headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=60) as r:
            return json.loads(r.read())["embedding"]

npc_id = sys.argv[1] if len(sys.argv) > 1 else "cat"
query = sys.argv[2] if len(sys.argv) > 2 else "玩家 最近做过什么 威胁 击杀 老鼠 锄地 浇水 收获 作物 商店"

# ---- 1. 当前 NPC 数据设置 ----
personas = json.load(open("npc_personas.json", encoding="utf-8"))["npcs"]
p = personas[npc_id]
print("=" * 62)
print(f"[1] NPC 数据设置: {p['display_name']} ({npc_id})")
print(f"    role_line = {p['role_line']}")
print(f"    memory    = {p['memory']}")
print(f"    inference = {p['inference']}")
print(f"    claims    = {p['claims']}")
print(f"    preset_vectors = {len(p.get('preset_vectors', []))} 条")

# ---- 2. 加载真实向量库 + 嵌入器, 捞玩家事件向量 ----
store = LogVectorStore("log_vector_store.jsonl")
print(f"[2] 向量库: {store.count()} 条事件向量 (log_vector_store.jsonl)")
nm = NpcMemory(store, personas, embedder=OllamaBgeM3(), actor="system")

recalled = nm.recall(npc_id, query)
facts = recalled.get("items", [])
print("    召回明细: 入选", len(facts), "| 门控丢弃", recalled.get("dropped_by_gate"), "| 阵营拒看", recalled.get("denied_by_access"))
print("    memory_block:")
for ln in (recalled.get("memory_block") or "").splitlines():
    print("      |", ln)
print(f"[3] CEL 召回(门1阵营访问 + 门2 conf>={p['memory']['confidence_threshold']}): {len(facts)} 条入选")
evidence_lines = []
for i, f in enumerate(facts[:p["memory"]["recall_top_k"]], 1):
    line = f"事实{i}[{f.get('source_type','未知')}/{f.get('confidence',0):.2f}]: {f.get('text','')}"
    print("    " + line)
    evidence_lines.append(line)
evidence_text = "\n".join(evidence_lines) if evidence_lines else "（无足信事实）"

# ---- 3. 组装推理 prompt (框架模板) ----
best_sim = max((f.get("confidence", 0.0) for f in facts), default=0.0)
prompt = build_inference_prompt(p, {"text": query, "sim": best_sim}, evidence_text, old_attitude="中性")
print("=" * 62)
print("[4] 组装好的推理 prompt:")
print("-" * 62)
print(prompt)
print("-" * 62)

# ---- 4. 调 qwen 生成推理结论 (0.8b 项目基线 + 7b 对比) ----
for model in ["qwen3.5:0.8b"]:
    try:
        out, dt = ollama_chat(model, prompt)
        print(f"[5] {model} 推理结论 ({dt:.1f}s):")
        print(out)
        conf = re.search(r"(?:置信度|confidence)[^\d]*(\d+(?:\.\d+)?)", out)
        c = float(conf.group(1)) / (100 if conf and float(conf.group(1)) > 1 else 1) if conf else None
        # grounded 不该信模型自述, 按框架语义=结论有没有足信事实支撑
        g = (len(facts) > 0) and (c is not None)
        verdict = "通过" if (c is not None and g is not None and c >= p["inference"]["base_confidence"] and g) else "未过/未解析"
        print(f"    >> 门控 (conf>={p['inference']['base_confidence']} && grounded): conf={c} grounded={g} → {verdict}")
        print("-" * 62)
    except Exception as e:
        print(f"[5] {model} 调用失败: {e}")
print("[i] mock 演示只读, 未写回向量库; 正式链路走 nm.store_inference(CEL 门控入库)")
