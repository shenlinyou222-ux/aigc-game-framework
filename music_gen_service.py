# -*- coding: utf-8 -*-
"""音乐生成服务(单例): 情绪槽位 -> NotaGen三条件 -> ABC 产物, 串行执行。

设计:
- 单实例锁(端口52773): NotaGen fp16 装载占 ~3.1GB 显存, 严禁并发;
- 任务投递: 往 jobs 目录丢 JSON 文件即可(voice_intent 只写文件, 不 Popen 模型);
- 串行消费: 一次只跑一个 notagen_gen 子进程, 产物写 gen_music/<slot>_<ts>.abc;
- 情绪槽位表(Phase2 曲库): 晴/雨/夜/紧张/温馨 -> (period, composer, instrumentation)。

用法:
  python music_gen_service.py            (常驻消费)
  echo '{"slot":"rain"}' > gen_music_jobs/job_x.json   (投递)
"""
import glob
import json
import os
import socket
import subprocess
import sys
import time

BASE = os.path.dirname(os.path.abspath(__file__))
JOBS_DIR = os.path.join(BASE, "gen_music_jobs")
OUT_DIR = os.path.join(BASE, "gen_music")
NOTAGEN_RUNNER = os.path.join(BASE, "notagen_gen.py")

# 情绪槽位 -> NotaGen 三条件(period=时期, composer=作曲家, instr=配器)
# 曲风直觉: 晴=轻快古典, 雨=浪漫钢琴, 夜=印象派, 紧张=现代, 温馨=浪漫弦乐
SLOTS = {
    "day":    ("Classical", "Mozart", "Piano"),
    "rain":   ("Romantic", "Chopin", "Piano"),
    "night":  ("Impressionist", "Debussy", "Piano"),
    "tense":  ("Modern", "Stravinsky", "Orchestra"),
    "cozy":   ("Romantic", "Schumann", "Strings"),
}

# 语音文本 -> 槽位(voice_intent 侧不重复维护, 都查这里)
SLOT_KEYWORDS = [
    (r"雨|阴|忧郁|难过", "rain"),
    (r"夜|晚|安静|睡", "night"),
    (r"紧张|战斗|激烈|刺激", "tense"),
    (r"温馨|治愈|放松|舒缓", "cozy"),
    (r"晴|白天|轻快|开心|欢快", "day"),
]

INSTR_MAP = {"钢琴": "Piano", "小提琴": "Violin", "吉他": "Guitar",
             "管弦": "Orchestra", "弦乐": "Strings"}


def text_to_job(text: str):
    """闲聊/指令文本 -> job dict(槽位或显式三条件)。识别不出返回 None。"""
    import re
    for pat, slot in SLOT_KEYWORDS:
        if re.search(pat, text):
            p, c, i = SLOTS[slot]
            return {"slot": slot, "period": p, "composer": c, "instr": i}
    m = re.search(r"钢琴|小提琴|吉他|管弦|弦乐", text)
    if m:
        return {"slot": "custom", "period": "Romantic", "composer": "Schumann",
                "instr": INSTR_MAP[m.group()]}
    return None


def _lock_or_exit():
    lock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        lock.bind(("127.0.0.1", 52773))
    except OSError:
        print("[music-gen] 已有实例在运行(端口52773), 退出")
        return False
    return True


def _run_job(job: dict) -> str:
    slot = job.get("slot", "custom")
    period = job.get("period") or SLOTS.get(slot, SLOTS["day"])[0]
    composer = job.get("composer") or SLOTS.get(slot, SLOTS["day"])[1]
    instr = job.get("instr") or SLOTS.get(slot, SLOTS["day"])[2]
    out = os.path.join(OUT_DIR, f"{slot}_{int(time.time())}.abc")
    print(f"[music-gen] RUN {slot}: {period}/{composer}/{instr} -> {os.path.basename(out)}",
          flush=True)
    t0 = time.time()
    r = subprocess.run(
        [sys.executable, "-X", "utf8", NOTAGEN_RUNNER, period, composer, instr, out],
        cwd=BASE, capture_output=True, text=True, encoding="utf-8", errors="replace",
        creationflags=0x4000)   # BELOW_NORMAL_PRIORITY_CLASS: 低优先级, 让路 whisper
    ok = r.returncode == 0 and os.path.exists(out)
    print(f"[music-gen] {'OK' if ok else 'FAIL'} {time.time()-t0:.1f}s"
          f"{'' if ok else ' :: ' + (r.stdout or r.stderr)[-300:]}", flush=True)
    return out if ok else ""


def main():
    if not _lock_or_exit():
        return
    os.makedirs(JOBS_DIR, exist_ok=True)
    os.makedirs(OUT_DIR, exist_ok=True)
    print(f"[music-gen] watching {JOBS_DIR}")
    while True:
        for job_file in sorted(glob.glob(os.path.join(JOBS_DIR, "*.json"))):
            try:
                with open(job_file, encoding="utf-8") as f:
                    job = json.load(f)
                _run_job(job)
            except Exception as e:
                print(f"[music-gen] job error {job_file}: {e}", flush=True)
            finally:
                try:
                    os.remove(job_file)   # 消费即删(成败都删, 不重试卡队列)
                except OSError:
                    pass
        time.sleep(2)


if __name__ == "__main__":
    main()
