# -*- coding: utf-8 -*-
"""上下文表情触发器: 监听游戏遥测事件流 -> 规则映射 + 向量召回 -> 弹表情
触发示例:
  shop.gift_claimed / story_dialog.gift_given -> 得到礼物
  story_dialog.finished                      -> 比心感谢
  story_dialog.cancelled                     -> 无奈叹气
  npc.sold_out                               -> 冷笑蔑视
语义层: 规则未命中的 command.executed 事件, 用向量库召回对比触发描述,
  相似度过阈值也触发(如 "获得/奖励/礼物"类事件)。
"""
import json, os, sys, time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import sticker_finder
from hot_load_pack import EVENTS, refresh_store, _store

RULES = {
    "shop.gift_claimed": "得到礼物",
    "story_dialog.gift_given": "得到礼物",
    "story_dialog.finished": "比心感谢",
    "story_dialog.cancelled": "无奈叹气",
    "npc.sold_out": "冷笑蔑视",
}
SEMANTIC_TRIGGERS = [
    ("获得 得到 礼物 奖励 道具 赠送", "得到礼物", 0.30),
    ("失败 错误 失败 拒绝 丢弃", "疑惑不解", 0.35),
]


def read_new_lines() -> list[str]:
    """增量读 game_events.jsonl(文件被游戏重写时自动重置偏移)。"""
    global _offset, _size
    try:
        size = os.path.getsize(EVENTS)
    except OSError:
        return []
    if size < _offset:          # 游戏重启重写了文件
        _offset = 0
    with open(EVENTS, encoding="utf-8", errors="ignore") as f:
        f.seek(_offset)
        lines = [l for l in f if l.strip()]
        _offset = f.tell()
    return lines


_offset, _size = 0, 0
_seen: set = set()


def semantic_match(event: dict) -> str | None:
    try:
        vs = _store()
        vs.add(event)   # 事件先入向量库(权威互动史)
        text = json.dumps(event.get("data", {}), ensure_ascii=False)
        for desc, emo, thr in SEMANTIC_TRIGGERS:
            r = vs.recall_hybrid(text, limit=1)
            hits = r.hits if hasattr(r, "hits") else r
            for h in hits:
                d = h.data if hasattr(h, "data") else h
                s = getattr(h, "score", 0) or 0
                if str(d).find("gift") >= 0 and emo == "得到礼物":
                    return emo   # 命中礼物类历史事实
    except Exception:
        pass
    return None


def main():
    # 单实例锁: 与 voice_intent 同机制
    import socket
    lock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        lock.bind(("127.0.0.1", 52772))
    except OSError:
        print("[ctx-sticker] 已有实例在运行, 退出")
        return
    # 播种: 已存在事件全部标记为已处理(只对新增事件反应)
    global _offset
    if os.path.exists(EVENTS):
        _offset = os.path.getsize(EVENTS)
    print("[ctx-sticker] watching", EVENTS)
    while True:
        for line in read_new_lines():
            try:
                ev = json.loads(line)
            except json.JSONDecodeError:
                continue
            eid = ev.get("event_id", "")
            if not eid or eid in _seen:
                continue
            _seen.add(eid)
            et = ev.get("event_type", "")
            emo = RULES.get(et)
            if emo is None and et == "command.executed":
                emo = semantic_match(ev)
            if emo:
                path = sticker_finder.show(emo, seconds=2.2)
                print(f"[ctx-sticker] {et} -> {emo} -> {path}")
        time.sleep(1.0)


if __name__ == "__main__":
    main()
