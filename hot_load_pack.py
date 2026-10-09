# -*- coding: utf-8 -*-
"""剧情资源包实时热加载器
包格式见 packs/*.json。加载语义:
  1. 前置命令(play_music 等)立即入队
  2. story_dialog 入队并等待游戏遥测确认 finished(轮询日志)
  3. after 命令(set_dialog 等)在剧情播完后入队 —— 尊重共享生命周期
幂等: 命令 command_id 由游戏侧 executed.json 全局去重, 重复加载同一包 = 全部 skip。
用法: python hot_load_pack.py packs/pack_cat_secret_base_v1.json [--dry]
"""
import json, os, sys, time

INBOX = os.path.join(os.environ.get("APPDATA") or os.path.expanduser("~/AppData/Roaming"), "Godot", "app_userdata",
                     "StardewValley", "commands", "inbox.jsonl")
EVENTS = os.path.join(os.environ.get("APPDATA") or os.path.expanduser("~/AppData/Roaming"), "Godot", "app_userdata",
                      "StardewValley", "logs", "game_events.jsonl")
STORE = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                     "dsh_context_store_bge3.jsonl")   # 权威互动数据源(持久)


def _store():
    import sys
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from aigc_core.log_vector_store import LogVectorStore
    return LogVectorStore(STORE)


def refresh_store() -> int:
    """增量 ingest 游戏事件流进向量库(权威源), 返回新增条数。"""
    vs = _store()
    return vs.ingest_jsonl(EVENTS)


def submit(cmd: dict) -> None:
    os.makedirs(os.path.dirname(INBOX), exist_ok=True)
    with open(INBOX, "a", encoding="utf-8") as f:
        f.write(json.dumps(cmd, ensure_ascii=False) + "\n")


def wait_event(substr: str, event_type_substr: str, timeout: float = 30.0) -> bool:
    """在权威事件流(game_events.jsonl, 逐条flush)里等包含子串的事件出现。"""
    t0 = time.time()
    while time.time() - t0 < timeout:
        try:
            with open(EVENTS, encoding="utf-8", errors="ignore") as f:
                for line in f:
                    if substr in line and event_type_substr in line:
                        return True
        except OSError:
            pass
        time.sleep(1.0)
    return False


def wait_finished(story_cmd_id: str, timeout: float = 60.0) -> bool:
    """轮询 game_events.jsonl(逐条flush)。重复包会 skip(无started事件), 10s 内
    未 started 则查 godot.log 的 skip duplicate 判定直接放行。"""
    t0 = time.time()
    seen_started = False
    while time.time() - t0 < timeout:
        try:
            with open(EVENTS, encoding="utf-8", errors="ignore") as f:
                events = [json.loads(l) for l in f if l.strip()]
        except OSError:
            events = []
        started = any(e["event_type"] == "story_dialog.started" and e["unix_time"] >= t0 - 5
                      for e in events)
        if started:
            seen_started = True
        finished = any(e["event_type"] == "story_dialog.finished" and e["unix_time"] >= t0
                       for e in events)
        if started and finished:
            return True
        if not seen_started and time.time() - t0 > 10.0:
            # 未 started: 大概率是重复包被游戏 skip -> 无需等待
            return False
        time.sleep(1.0)
    return False


def pack_exit_check(pack: dict, timeout: float = 120.0) -> str:
    """剧情包退出判定(以向量库为权威源):
    done    = 全部命令已执行 且(若带gift)礼物已被玩家真实领取(shop.gift_claimed)
    partial = 命令已执行但礼物未领(玩家还没确认) —— 等待
    timeout = 超时
    """
    import time as _t
    t0 = _t.time()
    ids = [c["command_id"] for c in pack["commands"]]
    gift_ids = set()
    for c in pack["commands"]:
        g = c.get("params", {}).get("gift", {})
        if c["command_type"] == "story_dialog" and g.get("item_id"):
            gift_ids.add(g["item_id"])
    while _t.time() - t0 < timeout:
        refresh_store()
        vs = _store()
        executed = set()
        claimed = set()
        for rec in vs.all():
            et = getattr(rec, "event_type", "")
            ut = getattr(rec, "unix_time", 0)
            # 只认本次加载之后的事实: 历史 gift_claimed/executed 不算数
            if ut < t0 - 5:
                continue
            if et == "command.executed":
                executed.add(str(getattr(rec, "data", {}).get("command_id", "")))
            elif et == "shop.gift_claimed":
                claimed.add(str(getattr(rec, "data", {}).get("item_id", "")))
        if all(i in executed for i in ids):
            if not gift_ids or gift_ids <= claimed:
                return "done"
            if _t.time() - t0 > 60:   # 礼物60s未领, 视为玩家暂离
                return "partial(gift_unclaimed)"
        _t.sleep(2.0)
    return "timeout"


def load_pack(path: str, dry: bool = False) -> None:
    pack = json.load(open(path, encoding="utf-8"))
    pid = pack["pack_id"]
    cmds = pack["commands"]
    print(f"== 加载剧情包 {pid} ({len(cmds)} 命令) ==")

    story_ids, afters = [], []
    for cmd in cmds:
        if cmd["command_type"] == "story_dialog":
            story_ids.append(cmd["command_id"])
            if not dry:
                submit(cmd)
            print(f"[queued ] {cmd['command_id']} ({cmd['command_type']})")
        elif cmd["command_type"] == "set_dialog":
            afters.append(cmd)          # 剧情生命周期约束: 播完再投
            print(f"[defer  ] {cmd['command_id']} (等 story finished)")
        else:
            if not dry:
                submit(cmd)
            print(f"[queued ] {cmd['command_id']} ({cmd['command_type']})")

    if story_ids and not dry:
        story_cmd = next(c for c in cmds if c["command_type"] == "story_dialog")
        tmo = 300.0 if story_cmd.get("params", {}).get("trigger") == "interact" else 60.0
        ok = wait_finished(story_ids[-1], timeout=tmo)
        print(f"[story  ] played={ok} ({'新剧情已播完' if ok else '重复包/未启动, 立即放行'})")
        for cmd in afters:
            submit(cmd)
            print(f"[queued ] {cmd['command_id']} (after-story)")
        # 退出机制: 以向量库权威互动数据判定包生命周期收口
        state = pack_exit_check(pack, timeout=tmo)
        # 回滚: 包退出后 NPC 恢复剧情前的原始对话(after 台词只是余韵, 不应成为终态)
        if state == "done" and pack.get("meta", {}).get("rollback", False):
            rb_id = f"{pid}_rollback"
            submit({"command_id": rb_id, "command_type": "reset_dialog",
                    "actor": "pack", "params": {"npc": pack["meta"].get("npc", "Cat")}})
            print(f"[queued ] {rb_id} (reset_dialog 回滚)")
            if wait_event(f'"command_id":"{rb_id}"', '"command.executed"'):
                state = "done+rolled_back"
                print(f"[rollback] {pid} 原始对话已恢复")
        marker = os.path.join(os.path.dirname(os.path.abspath(__file__)), "packs", ".state")
        os.makedirs(marker, exist_ok=True)
        with open(os.path.join(marker, f"{pid}.{state.split('(')[0]}"), "w", encoding="utf-8") as f:
            f.write(f"{pid}\t{state}\t{time.strftime('%Y-%m-%dT%H:%M:%S')}\n")
        print(f"[exit   ] {pid} -> {state} (marker written)")
    elif dry:
        print("(dry-run, 未提交)")
    print("== 加载完成 ==")


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if a != "--dry"]
    load_pack(args[0] if args else "packs/pack_cat_secret_base_v1.json",
              dry="--dry" in sys.argv)
