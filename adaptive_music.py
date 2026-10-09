# -*- coding: utf-8 -*-
"""自适应切歌聚合层: 情绪+遥测 -> 槽位, 达门槛才投 play_music。

优先级:
  显式指令(用户喊 下雨/晴天/换一首) —— 立即生效, 由 voice_intent 直接投 inbox;
  本服务只做隐式跟随: 三重门槛(单条置信/时间窗持续性/滞回冷却) + 专注期不切。

数据源: game_events.jsonl
  chat.emotion            情绪打分(dict/semantic/default)
  music.switched          显式切歌已发生 -> 同步槽位+刷冷却
  input.action/tool_used  行为频率 -> 专注期保护

投递: inbox.jsonl play_music(经 command_server 执行, 复用交叉淡入)
"""
import glob
import json
import os
import re
import socket
import time

EVENTS = os.path.join(os.environ.get("APPDATA") or os.path.expanduser("~/AppData/Roaming"), "Godot", "app_userdata",
                      "StardewValley", "logs", "game_events.jsonl")
INBOX = os.path.join(os.environ.get("APPDATA") or os.path.expanduser("~/AppData/Roaming"), "Godot", "app_userdata",
                     "StardewValley", "commands", "inbox.jsonl")

# 情绪 -> 音乐槽位(与 music_gen_service.SLOTS 对齐: day/rain/night/tense/cozy)
EMO_TO_SLOT = {
    "开心对视": "day", "点赞加油": "day", "自信得意": "day", "得到礼物": "day",
    "大哭流泪": "rain",
    "生气发怒": "tense", "坏笑": "tense",
    "疑惑不解": "night", "已然睡着": "night", "流汗尴尬": "night",
    "无奈叹气": "cozy", "可爱卖萌": "cozy", "比心感谢": "cozy",
}
TRACK_OF_SLOT = {"day": "day", "rain": "rain2", "night": "night",
                 "tense": "tense", "cozy": "cozy"}   # rain槽位映射雨歌rain2; 环境雨声归Ambient并行通道

# 门槛参数
MIN_SCORE = 0.6          # ① 单条置信: 只采信高分事件
WINDOW_SEC = 150         # ② 时间窗: 近2.5分钟
MAJORITY = 0.6           # ② 主情绪占比门槛
COOLDOWN_SEC = 300       # ③ 滞回: 切歌后5分钟不再隐式切
USER_ACTIVE_SEC = 30     # 用户近30秒发过显式指令/语音 -> 隐式让路, 不抢切
ACTION_LIMIT = 25        # 专注期保护: 窗口内动作数超此值不切

_pos = 0
_current_slot = "day"
_last_switch = 0.0
_last_user_active = 0.0   # 最近一次用户语音/显式指令时间
_actions = []       # [(ts)] 行为动作
_emos = []          # [(ts, slot, score)] 已采信情绪


def _tail_events():
    """增量读 game_events.jsonl, 返回新行列表。"""
    global _pos
    out = []
    try:
        size = os.path.getsize(EVENTS)
        if size < _pos:
            _pos = 0   # 日志被轮转/清空
        with open(EVENTS, encoding="utf-8") as f:
            f.seek(_pos)
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    out.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
            _pos = f.tell()
    except FileNotFoundError:
        pass
    return out


def _prune(now):
    global _actions, _emos
    _actions = [t for t in _actions if now - t <= 60]
    _emos = [(t, s, sc) for t, s, sc in _emos if now - t <= WINDOW_SEC]


def _submit_switch(slot: str, reason: str):
    global _current_slot, _last_switch
    track = TRACK_OF_SLOT.get(slot)
    if not track:
        return   # night/tense/cozy 槽位曲库还没渲染出来, 暂不投
    # 切歌进行中提示: 弹专用表情包告知用户(被动切换用户无预期, 更需要提示)
    try:
        import sticker_finder
        sticker_finder.show("自信得意", seconds=1.5)
    except Exception:
        pass
    cmd = {"command_id": f"adaptive_{int(time.time()*1000)}",
           "command_type": "play_music", "actor": "adaptive_music",
           "params": {"track": track}}
    with open(INBOX, "a", encoding="utf-8") as f:
        f.write(json.dumps(cmd, ensure_ascii=False) + "\n")
    _current_slot, _last_switch = slot, time.time()
    print(f"[adaptive-music] SWITCH -> {track} ({reason})", flush=True)


def _implicit_check(now):
    """隐式跟随: 三重门槛全过才切。"""
    if now - _last_switch < COOLDOWN_SEC:
        return
    if now - _last_user_active < USER_ACTIVE_SEC:
        return   # 用户在主动说话/下指令, 隐式让路不抢切
    if len(_actions) > ACTION_LIMIT:
        return   # 专注期不切
    if not _emos:
        return
    votes = {}
    for _, slot, sc in _emos:
        votes[slot] = votes.get(slot, 0.0) + sc   # 票重 = score
    total = sum(votes.values())
    if total <= 0:
        return
    top, w = max(votes.items(), key=lambda kv: kv[1])
    if w / total >= MAJORITY and top != _current_slot:
        _submit_switch(top, f"implicit {w/total:.0%} over {len(_emos)} events")


def main():
    lock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        lock.bind(("127.0.0.1", 52774))
    except OSError:
        print("[adaptive-music] 已有实例在运行(端口52774), 退出")
        return
    print(f"[adaptive-music] watching {EVENTS}")
    global _current_slot, _last_switch, _last_user_active
    while True:
        now = time.time()
        for ev in _tail_events():
            et, data = ev.get("event_type"), ev.get("data", {})
            ts = ev.get("unix_time", now)
            if et == "chat.emotion":
                sc = data.get("score", 0.0)
                slot = EMO_TO_SLOT.get(data.get("emotion", ""))
                if slot and sc >= MIN_SCORE:
                    _emos.append((ts, slot, sc))
            elif et == "music.switched":
                # 显式切歌(用户指令或本服务投递的落地点): 同步槽位+刷冷却
                track = data.get("track", "")
                _current_slot = track if track in TRACK_OF_SLOT else _current_slot
                _last_switch = max(_last_switch, ts)
                _last_user_active = max(_last_user_active, ts)
            elif et == "voice.captured":
                _last_user_active = max(_last_user_active, ts)   # 用户在说话, 隐式让路
            elif et in ("input.action", "player.tool_used"):
                _actions.append(ts)
        _prune(now)
        _implicit_check(now)
        time.sleep(3)


if __name__ == "__main__":
    main()
