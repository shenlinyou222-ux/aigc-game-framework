# -*- coding: utf-8 -*-
"""三段式台词热修改管线:
  1. qwen3.5 (Ollama) 产出 YAML 文本
  2. 确定性处理: 解析/校验/清洗 -> set_dialog JSON 命令
  3. 提交 Godot inbox.jsonl 热替换 NPC 台词 (无需重启游戏)

用法: python hot_dialog_pipeline.py [npc] [style]
"""
import json, os, re, sys, time, urllib.request

OLLAMA = "http://127.0.0.1:11434/api/chat"
MODEL = "qwen3.5:0.8b"
INBOX = os.path.join(os.environ.get("APPDATA") or os.path.expanduser("~/AppData/Roaming"), "Godot", "app_userdata",
                     "StardewValley", "commands", "inbox.jsonl")
MAX_CHARS = 15          # 单句硬上限(游戏对话框约束)
MAX_LINES = 5


# ── 阶段1: 模型产出 YAML ─────────────────────────────────────────
def gen_yaml(npc: str, style: str, beats: list[str]) -> str:
    tpl = "\n".join(f"  - beat: {b}" for b in beats)
    sys_p = (
        "你是游戏台词生成器。严格按照下面的示例格式输出, 只输出这个 YAML, 不要代码块, 不要解释:\n"
        "npc: 猫\n"
        "lines:\n"
        "  - 哼, 才不是在等你\n"
        "  - 摸头也也不是不行\n"
        "  - 陪本喵晒太阳啦\n"
        "注意: lines 下每一项就是一个 `- ` 开头的纯字符串台词, 每句不超过%d字。" % MAX_CHARS
    )
    user_p = f"角色: {npc}, 人设: {style}\n剧情节拍(按顺序每拍一句台词):\n{tpl}"
    body = {"model": MODEL, "messages": [
                {"role": "system", "content": sys_p},
                {"role": "user", "content": user_p}],
            "stream": False, "think": False, "options": {"temperature": 0.85}}
    req = urllib.request.Request(OLLAMA,
        data=json.dumps(body, ensure_ascii=False).encode("utf-8"),
        headers={"Content-Type": "application/json"})
    return json.load(urllib.request.urlopen(req, timeout=180))["message"]["content"]


# ── 阶段2: 确定性处理 YAML -> JSON 命令 ───────────────────────────
def _parse_yaml_light(text: str) -> tuple[str, list[str]]:
    """防御式解析: 主路径 YAML(npc:/lines:/- 项), 兜底 JSON 变体(text 字段)。"""
    # 兜底1: 整体就是 JSON
    s0 = text.strip().strip("`")
    if s0.startswith("{"):
        try:
            d = json.loads(s0)
            npc = str(d.get("npc", ""))
            lines = []
            for it in d.get("lines", []):
                if isinstance(it, str):
                    lines.append(it)
                elif isinstance(it, dict):
                    lines.append(str(it.get("text") or it.get("line") or next(iter(it.values()), "")))
            return npc, lines
        except json.JSONDecodeError:
            pass
    # 兜底2: 夹杂的 "text": "..." 键值对
    if '"text"' in text and not re.search(r"^lines:", text, re.M):
        return "", re.findall(r'"text"\s*:\s*"([^"]+)"', text)
    # 主路径: 轻量 YAML
    npc, lines, in_lines = "", [], False
    for raw in text.splitlines():
        line = raw.rstrip()
        s = line.strip()
        if not s or s.startswith("```"):
            continue
        if s.startswith("npc:"):
            npc = s.split(":", 1)[1].strip().strip("'\""); in_lines = False
        elif s.startswith("lines:"):
            in_lines = True
        elif in_lines and (s.startswith("- ") or s == "-"):
            lines.append(s[2:].strip().strip("'\""))
    # 兜底3: 全文只有裸 "- " 列表项(模型忘了包 npc/lines 键)
    if not lines and not npc:
        bare = [l.strip()[2:].strip().strip("'\"")
                for l in text.splitlines() if l.strip().startswith("- ")]
        if bare:
            return "", bare
    return npc, lines


def sanitize(lines: list[str]) -> list[str]:
    out = []
    for l in lines:
        l = re.sub(r"\s+", " ", l).strip()
        l = l.strip("。！!.")                       # 对话框不需要句读尾巴
        if not l:
            continue
        if len(l) > MAX_CHARS:
            l = l[:MAX_CHARS]                      # 硬截断, 确定性优于拒绝
        out.append(l)
        if len(out) >= MAX_LINES:
            break
    if not out:
        raise ValueError("模型输出无可台词, 原文见日志")
    return out


def yaml_to_command(yaml_text: str, npc_fallback: str, actor: str) -> dict:
    npc, lines = _parse_yaml_light(yaml_text)
    npc = npc or npc_fallback
    lines = sanitize(lines)
    return {"command_id": f"hot_dialog_{int(time.time()*1000)}",
            "command_type": "set_dialog", "actor": actor,
            "params": {"npc": npc, "lines": lines}}


# ── 阶段3: 提交热替换 ────────────────────────────────────────────
def submit(cmd: dict) -> str:
    os.makedirs(os.path.dirname(INBOX), exist_ok=True)
    with open(INBOX, "a", encoding="utf-8") as f:
        f.write(json.dumps(cmd, ensure_ascii=False) + "\n")
    return INBOX


if __name__ == "__main__":
    npc = sys.argv[1] if len(sys.argv) > 1 else "Cat"
    style = sys.argv[2] if len(sys.argv) > 2 else "傲娇又爱撒娇的猫"
    beats = ["第一次被玩家搭话, 嘴硬地打招呼", "其实想被摸, 不好意思地说", "邀请玩家一起玩"]

    y = ""
    for attempt in range(2):   # 确定性重试策略: 失败重采样一次
        y = gen_yaml(npc, style, beats)
        print("== 阶段1 qwen YAML (attempt %d) ==\n%s" % (attempt + 1, y.strip()))
        try:
            cmd = yaml_to_command(y, npc, MODEL)
            break
        except ValueError:
            if attempt == 1:
                raise
    print("\n== 阶段2 确定性转换 JSON ==\n" + json.dumps(cmd, ensure_ascii=False, indent=2))
    p = submit(cmd)
    print(f"\n== 阶段3 已提交热替换 -> {p}")
