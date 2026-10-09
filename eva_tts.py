# -*- coding: utf-8 -*-
"""EVA 机械女声 TTS 管线: edge-tts(在线) / SAPI(兜底) -> numpy 机械感 DSP -> 游戏资产缓存.

- 唯一 AI 语音角色 = EVA(机械女声); 其余剧情角色只有字幕, 不配音;
- 产物: experiment/gdgs_lab/assets/eva/E_<key>.wav + manifest.json (text -> wav);
- 游戏侧 _eva_announce 按整句文本查 manifest, 命中即播缓存, 未命中走 SAPI 实时兜底;
- 机械感 = 轻度环形调制(57Hz) + 电台带通(130-6800Hz) + 双梳状延迟 + 软削波.

用法: python eva_tts.py            # 全量(重)合成
      python eva_tts.py --keys l1_start finale_answer   # 指定行
"""
import asyncio
import json
import os
import subprocess
import sys

import numpy as np
import soundfile as sf
from scipy.signal import butter, sosfilt

BASE = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(BASE, "..", "experiment", "gdgs_lab", "assets", "eva")
VOICE = "zh-CN-XiaoxiaoNeural"
RATE = "+8%"
PITCH = "-3Hz"

# ===== 台词单一事实源(key -> 文本) —— 游戏 solar_combat.gd 同句引用 =====
LINES = {
    "l1_start": "EVA 在线。第一关：拦截并歼灭所有来犯敌机。",
    "l1_clear": "敌机群已歼灭。中转站核心已投放，返航进入即可跃迁。",
    "wormhole": "虫洞跃迁开始。",
    "l2_start": "第二关：保卫雷达相阵。敌方将持续空袭。",
    "l2_story1": "这里是前哨 K-7 空域。敌方主力正扑向雷达相阵。",
    "l2_story2": "守住它。增援已经在路上了。",
    "l2_reinf": "检测到集群跳跃信号——增援中队抵达战区。",
    "l2_reinf_eva": "增援中队已进入护航队列。集中火力，保卫雷达阵。",
    "fac_60": "警告：雷达阵结构完整度百分之六十。",
    "fac_30": "警告：雷达阵濒临损毁！",
    "l2_clear": "防御战达成。中转站核心已投放，返航跃迁。",
    "l3_start": "最终战。巨型目标出现，轨道炮台支援充能中。",
    "boss_warn": "警告：巨型目标正在接近！",
    "l3_support": "轨道炮台充能完毕——周期火力支援上线。",
    "l3_boss_down": "目标已被摧毁。……侦测到超巨型生物信号接近。",
    "whale_seen": "深空巨兽正在苏醒。",
    "whale_line1": "这段歌声……不在任何已知物种的声纹库中。",
    "finale_answer": "是它们在哀悼死去的同类。",
    "demo_complete": "全任务完成。向您致敬，指挥官。",
    "fail": "任务失败。按 R 重新部署。",
    "ack": "EVA 在线。说 EVA 查询当前任务 试试。",
}


def robotic(x: np.ndarray, sr: int) -> np.ndarray:
    """机械女声后处理: 环形调制 + 电台带通 + 梳状延迟 + 软削波。"""
    t = np.arange(len(x), dtype=np.float64) / sr
    x = x * (1.0 - 0.16 + 0.16 * np.sin(2 * np.pi * 57.0 * t))          # 轻度金属质感
    x = sosfilt(butter(2, 130, "hp", fs=sr, output="sos"), x)           # 去隆隆
    x = sosfilt(butter(3, 6800, "lp", fs=sr, output="sos"), x)          # 电台带宽
    d1, d2 = int(0.019 * sr), int(0.037 * sr)                           # 舱内通讯梳状反射
    y = x.copy()
    y[d1:] += 0.20 * x[:-d1]
    y[d2:] += 0.09 * x[:-d2]
    peak = max(float(np.max(np.abs(y))), 1e-9)
    y = np.tanh(1.35 * y / peak) / np.tanh(1.35)
    return (y / max(float(np.max(np.abs(y))), 1e-9) * 0.82).astype(np.float32)


def synth_edge(text: str, tmp_mp3: str) -> bool:
    async def _go():
        import edge_tts
        c = edge_tts.Communicate(text, VOICE, rate=RATE, pitch=PITCH)
        await c.save(tmp_mp3)

    for _ in range(3):
        try:
            asyncio.run(_go())
            if os.path.exists(tmp_mp3) and os.path.getsize(tmp_mp3) > 1000:
                return True
        except Exception as e:
            print(f"  edge-tts retry: {e}", flush=True)
    return False


def synth_sapi(text: str, out_wav: str) -> bool:
    """离线兜底: Windows SAPI 中文嗓音(如 Huihui) -> wav, 后续走同一 DSP。"""
    ps = (
        "Add-Type -AssemblyName System.Speech; "
        "$s = New-Object System.Speech.Synthesis.SpeechSynthesizer; "
        "$zh = $s.GetInstalledVoices() | Where-Object { $_.VoiceInfo.Culture.Name -like 'zh*' } | Select-Object -First 1; "
        "if ($zh) { $s.SelectVoice($zh.VoiceInfo.Name) }; "
        f"$s.Rate = 2; $s.SetOutputToWaveFile('{out_wav}'); "
        f"$s.Speak('{text.replace(chr(39), chr(39)*2)}'); $s.Dispose()"
    )
    r = subprocess.run(["powershell", "-NoProfile", "-Command", ps],
                       capture_output=True, text=True, creationflags=0x08000000)
    return r.returncode == 0 and os.path.exists(out_wav) and os.path.getsize(out_wav) > 1000


def render(key: str, text: str, engine: str) -> str:
    """合成单行 -> 机械感 DSP -> wav; 返回文件名(失败返回空)。"""
    tmp_mp3 = os.path.join(OUT_DIR, f"_tmp_{key}.mp3")
    tmp_wav = os.path.join(OUT_DIR, f"_tmp_{key}.wav")
    ok = synth_edge(text, tmp_mp3) if engine == "edge" else synth_sapi(text, tmp_wav)
    if not ok and engine == "edge":
        print(f"  [{key}] edge 失败 -> SAPI 兜底", flush=True)
        ok = synth_sapi(text, tmp_wav)
    src = tmp_mp3 if os.path.exists(tmp_mp3) and os.path.getsize(tmp_mp3) > 1000 else tmp_wav
    if not ok:
        print(f"  [{key}] 全部引擎失败, 跳过", flush=True)
        return ""
    try:
        x, sr = sf.read(src, dtype="float32", always_2d=True)
        x = x.mean(axis=1)
        out = robotic(x, sr)
        name = f"E_{key}.wav"
        sf.write(os.path.join(OUT_DIR, name), out, sr, subtype="PCM_16")
        print(f"  [{key}] {len(out)/sr:.2f}s ({engine}) -> {name}", flush=True)
        return name
    finally:
        for p in (tmp_mp3, tmp_wav):
            if os.path.exists(p):
                try:
                    os.remove(p)
                except OSError:
                    pass


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    keys = sys.argv[sys.argv.index("--keys") + 1].split(",") if "--keys" in sys.argv else list(LINES)
    manifest_path = os.path.join(OUT_DIR, "manifest.json")
    manifest = {}
    if os.path.exists(manifest_path):
        with open(manifest_path, encoding="utf-8") as f:
            raw = json.load(f)
        manifest = raw.get("text_to_wav", raw) if isinstance(raw, dict) else {}
        if not all(isinstance(v, str) for v in manifest.values()):
            manifest = {}   # 结构损坏(旧版平铺/嵌套) -> 全量重建
    engine = "edge"
    for k in keys:
        if k not in LINES:
            print(f"  未知台词 key: {k}")
            continue
        name = manifest.get(LINES[k]) or render(k, LINES[k], engine)
        if name:
            manifest[LINES[k]] = name
    if "--manifest-only" not in sys.argv:
        pass
    # by_key: 游戏按台词 key 取 {text, wav} —— 单一事实源, 避免两侧文本漂移
    by_key = {k: {"text": t, "wav": manifest[t]} for k, t in LINES.items() if t in manifest}
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump({"text_to_wav": manifest, "by_key": by_key}, f, ensure_ascii=False, indent=1)
    print(f"manifest: {len(manifest)} 行 by_key {len(by_key)} -> {manifest_path}")


if __name__ == "__main__":
    main()
