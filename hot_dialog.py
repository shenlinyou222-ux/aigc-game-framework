# -*- coding: utf-8 -*-
"""qwen3.5 -> Godot NPC 台词热修改工具
用法:
  python hot_dialog.py Cat            # 默认傲娇猫
  python hot_dialog.py Mouse 温柔的鼠商人
链路: Ollama qwen3.5 逐句生成 -> inbox.jsonl set_dialog 命令 -> CommandServer 热替换 dialog
"""
import json, os, sys, time, urllib.request

try:
    from model_batch_manager import policy_keep_alive  # 非常驻策略(45s 空闲窗), 训练前自动腾卡
except ImportError:
    policy_keep_alive = lambda model=None: "45s"

OLLAMA = "http://127.0.0.1:11434/api/chat"
MODEL = "qwen3.5:0.8b"
INBOX = os.path.join(os.environ.get("APPDATA") or os.path.expanduser("~/AppData/Roaming"), "Godot", "app_userdata", "StardewValley", "commands", "inbox.jsonl")


def chat(user_prompt: str, style: str) -> str:
    body = {
        "model": MODEL,
        "messages": [
            {"role": "system", "content": f"为农场游戏的NPC写一句台词, 人设:{style}。只输出这一句, 不超过15个汉字, 不要引号不要解释。"},
            {"role": "user", "content": user_prompt},
        ],
        "stream": False, "think": False,
        "options": {"temperature": 0.9},
        "keep_alive": policy_keep_alive(MODEL),
    }
    req = urllib.request.Request(OLLAMA, data=json.dumps(body, ensure_ascii=False).encode("utf-8"),
                                 headers={"Content-Type": "application/json"})
    txt = json.load(urllib.request.urlopen(req, timeout=180))["message"]["content"]
    return txt.strip().strip('"').splitlines()[0]


def hot_swap(npc: str, style: str, beats: list[str]) -> list[str]:
    lines = [chat(b, style) for b in beats]
    cmd = {"command_id": f"hot_dialog_{int(time.time())}", "command_type": "set_dialog",
           "actor": MODEL, "params": {"npc": npc, "lines": lines}}
    os.makedirs(os.path.dirname(INBOX), exist_ok=True)
    with open(INBOX, "a", encoding="utf-8") as f:
        f.write(json.dumps(cmd, ensure_ascii=False) + "\n")
    return lines


if __name__ == "__main__":
    npc = sys.argv[1] if len(sys.argv) > 1 else "Cat"
    style = sys.argv[2] if len(sys.argv) > 2 else "傲娇又爱撒娇的猫"
    beats = ["第一次被玩家搭话, 嘴硬地打招呼", "其实想亲近, 不好意思地说", "邀请玩家一起玩"]
    print(hot_swap(npc, style, beats))
