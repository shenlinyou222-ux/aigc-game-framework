# -*- coding: utf-8 -*-
"""表情包检索器 v2: 游戏内精选目录, 文件名=情绪_角色_序号.png
文件名即唯一真相, 无需标签库。
"""
import os, random, re, json, time

STICKER_DIR = r"C:\Users\user\Desktop\ds harness\godot_valley_demo\stickers"
INBOX = os.path.join(os.environ.get("APPDATA") or os.path.expanduser("~/AppData/Roaming"), "Godot", "app_userdata",
                     "StardewValley", "commands", "inbox.jsonl")

_files = None


def index() -> list:
    global _files
    if _files is None:
        _files = [f for f in os.listdir(STICKER_DIR) if f.endswith(".png")]
    return _files


def find(emotion: str, character: str = None):
    """按情绪(必需)+角色(可选)检索, 返回绝对路径(正斜杠)。"""
    emo = emotion.strip()
    pool = []
    for f in index():
        parts = f[:-4].split("_")
        if parts[0] != emo:
            continue
        if character and (len(parts) < 2 or parts[1] != character):
            continue
        pool.append(f)
    if not pool:
        return None
    return (STICKER_DIR + "/" + random.choice(pool)).replace("\\", "/")


def show(emotion: str, character: str = None,
         seconds: float = 2.5, size: float = 0.3):
    path = find(emotion, character)
    if not path:
        return None
    cmd = {"command_id": f"sticker_{int(time.time()*1000)}",
           "command_type": "show_sticker", "actor": "context",
           "params": {"file": path, "seconds": seconds, "size": size}}
    with open(INBOX, "a", encoding="utf-8") as f:
        f.write(json.dumps(cmd, ensure_ascii=False) + "\n")
    return path


if __name__ == "__main__":
    import sys
    emo = sys.argv[1] if len(sys.argv) > 1 else "坏笑"
    print(f"共 {len(index())} 张;", emo, "->", find(emo))
