# -*- coding: utf-8 -*-
"""语音意图服务: 监听游戏 voice 目录 -> Qwen3-ASR 中文转写 -> 意图识别 -> inbox 命令
用法: python voice_intent.py          (前台常驻)
意图表(规则优先, 兜底可接qwen):
  下雨/雨天   -> play_ambient rain (环境雨声循环, BGM不中断)
  晴天/白天   -> play_ambient off (关环境音, BGM不中断)
  切歌/换一首 -> play_music 下一首(歌单只含歌曲, 环境音并行不受影响)
  剧情/讲故事 -> 加载最新剧情包(hot_load_pack)
  台词 <文本> -> set_dialog Cat
  铃铛/礼物   -> story_dialog 送铃铛
  快/慢一点   -> set_speed 移速倍率(0.35~2.5, 恢复正常=1.0)
  传送 <地名/坐标> -> tp (村口/出生点; "坐标63 2"小数字按瓦片*16+8换算)
  走去/带我去 <地名/NPC> -> walk_to 直线寻路(NPC实时位置, 停身旁24px)
  背包 <物品> -> find_item 背包查询(枚举物品数量/AI动态物品/总览)
  EVA 开头   -> 舰载语音播报(gdgs_lab): 查询当前任务/汇报战况/播报资源/帮助/再说一遍/安静
"""
import glob, json, os, re, subprocess, sys, time, urllib.request, uuid
from eva_intents import classify_eva, classify_eva_bare   # EVA voice-broadcast intents (gdgs_lab)

VOICE_DIR = os.path.join(os.environ.get("APPDATA") or os.path.expanduser("~/AppData/Roaming"), "Godot", "app_userdata",
                         "StardewValley", "voice")
INBOX = os.path.join(os.environ.get("APPDATA") or os.path.expanduser("~/AppData/Roaming"), "Godot", "app_userdata",
                     "StardewValley", "commands", "inbox.jsonl")
EVENTS = os.path.join(os.environ.get("APPDATA") or os.path.expanduser("~/AppData/Roaming"), "Godot", "app_userdata",
                      "StardewValley", "logs", "game_events.jsonl")
QWEN_ASR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        "models", "qwen3-asr-0.6b")   # Qwen3-ASR-0.6B-hf(本地, 短音频实测166ms/条)

_seen = set()
_last_track = "day"   # 切歌路由记忆(兜底): 真相读 game_events 里最后一次 music.switched

# 切歌轮播歌单(顺序即切换顺序), 同人曲为 NotaGen 生成渲染产物
# 只含歌曲; 环境音(rain 雨声)已拆到并行通道 play_ambient, 不再占歌单位
MUSIC_PLAYLIST = ["day", "night", "night2", "cozy", "cozy2",
                  "tense", "tense2", "custom1", "custom2", "day2", "rain2"]

# 表情提示: 切歌(被动/主动换曲)专用一张; 切天气按目标弹对应表情
SWITCHING_STICKER = "自信得意"          # 切歌进行中提示, 改名字即换图
WEATHER_STICKER = {"rain": "无奈叹气", "day": "开心对视"}

# ===== 玩家操控词表(移速/传送/寻路/背包) =====
# 地名 key 与游戏端 CommandServer.PLAYER_POIS 对齐(坐标事实源在游戏侧)
PLACE_WORDS = [
    (r"村口|村子|村庄|村中心|村里|集市|广场", "village"),
    (r"出生点|起点|出生的地方", "spawn"),
]
# 中文物品词 -> Enum.Item key(背包查询), 捕获原文作播报名; 与游戏端 Enum.Item 对齐
ITEM_WORDS = [
    (r"木头|木材|木柴|柴火", "WOOD"),
    (r"苹果", "APPLE"), (r"番茄|西红柿", "TOMATO"), (r"玉米", "CORN"),
    (r"小麦", "WHEAT"), (r"南瓜", "PUMPKIN"), (r"鱼", "FISH"),
]
# 移速词 -> 倍率(游戏端限幅0.2~4.0), 顺序即优先级
SPEED_WORDS = [
    (r"飞快|很快|最快|光速|风一样", 2.5),
    (r"快一?[点些]|加速|提速|加快|跑快", 1.6),
    (r"慢一?[点些]|减速|降速|放慢|跑慢", 0.6),
    (r"龟速|超慢|最慢", 0.35),
    (r"正常|恢复|还原", 1.0),
]
# 门控自动生成: 核心触发词(说了但没给程度, 走默认兜底) ∪ 全部程度词, 单一事实源永不同步
_SPEED_CORE = r"移速|一速|速度|变速|[走跑]得?快|[走跑]得?慢|快一?点|慢一?点|[调調][到为為]"
SPEED_GATE = re.compile("|".join([_SPEED_CORE] + [p for p, _ in SPEED_WORDS]))


_CUDA12_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           "models", "cuda12")   # torch升级CUDA13后ctranslate2仍需12版cublas
_on_cpu = False





_qw_model = _qw_proc = None


def _asr_qwen3(wav: str) -> str:
    """Qwen3-ASR-0.6B-hf: LLM解码, 短音频实测166ms/条; 强制中文"""
    global _qw_model, _qw_proc
    if _qw_proc is None:
        import torch
        from transformers import AutoProcessor, AutoModelForMultimodalLM
        _qw_proc = AutoProcessor.from_pretrained(QWEN_ASR)
        _qw_model = AutoModelForMultimodalLM.from_pretrained(QWEN_ASR, dtype="auto", device_map="cuda")
        _qw_model.eval()
    inputs = _qw_proc.apply_transcription_request(audio=wav, language="zh").to(_qw_model.device, _qw_model.dtype)
    ids = _qw_model.generate(**inputs, max_new_tokens=128)
    return str(_qw_proc.decode(ids[:, inputs["input_ids"].shape[1]:],
                               return_format="transcription_only")[0]).strip()


# ===== 主路后端: llama.cpp Qwen3-ASR Q4_K_M (纯CPU零显存, 常驻 8124, 实测0.1~0.3s/条) =====
# 服务/模型细节: tool_registry/stt_llamaserver.json; 失败自动回退下方 transformers GPU 后端
STT_URL = "http://127.0.0.1:8124"
STT_EXE = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       "models", "llamacpp", "llama-server.exe")
STT_CWD = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
_stt_spawn_ts = 0.0   # 上次拉起尝试时刻(60s 退避, 防止每条语音都重试拉起)


def _stt_alive(timeout: float = 0.6) -> bool:
    try:
        with urllib.request.urlopen(STT_URL + "/health", timeout=timeout) as r:
            return b"ok" in r.read()
    except Exception:
        return False


def _stt_ensure() -> bool:
    """健康检查; 服务不在则分离式拉起一次, 最多等 ~12s 模型加载。"""
    global _stt_spawn_ts
    if _stt_alive():
        return True
    if time.time() - _stt_spawn_ts < 60:
        return _stt_alive(timeout=2.0)   # 退避期内只探测不拉起
    _stt_spawn_ts = time.time()
    try:
        subprocess.Popen(
            [STT_EXE, "-m", "qwen3-asr-gguf-1.7b/Qwen3-ASR-1.7B-Q4_K_M.gguf",
             "--mmproj", "qwen3-asr-gguf-1.7b/mmproj-Qwen3-ASR-1.7B-Q4_K_M.gguf",
             "-c", "4096", "--port", "8124", "--no-webui"],
            cwd=STT_CWD,
            creationflags=0x00000008 | 0x00000200,   # DETACHED_PROCESS | NEW_PROCESS_GROUP
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        print("[voice] STT 服务未在运行, 已分离拉起 llama-server:8124")
    except OSError as e:
        print(f"[voice] STT 服务拉起失败: {e}")
        return False
    for _ in range(24):
        if _stt_alive():
            return True
        time.sleep(0.5)
    return False


def _asr_http(wav: str) -> str:
    """multipart POST 到 llama.cpp /v1/audio/transcriptions, 返回清洗后纯文本。"""
    boundary = uuid.uuid4().hex
    with open(wav, "rb") as f:
        audio = f.read()
    body = (
        f"--{boundary}\r\n"
        f"Content-Disposition: form-data; name=\"model\"\r\n\r\n"
        f"qwen3-asr\r\n"
        f"--{boundary}\r\n"
        f"Content-Disposition: form-data; name=\"file\"; filename=\"{os.path.basename(wav)}\"\r\n"
        f"Content-Type: audio/wav\r\n\r\n"
    ).encode() + audio + f"\r\n--{boundary}--\r\n".encode()
    req = urllib.request.Request(
        STT_URL + "/v1/audio/transcriptions", data=body, method="POST",
        headers={"Content-Type": f"multipart/form-data; boundary={boundary}"})
    with urllib.request.urlopen(req, timeout=30) as r:
        text = json.loads(r.read().decode()).get("text", "")
    return re.sub(r"^.*?<asr_text>", "", text).strip()


def transcribe(wav: str) -> str:
    """ASR 主路: llama.cpp Q4_K_M HTTP(零显存); 不可用自动回退 transformers GPU 后端。"""
    try:
        if _stt_ensure():
            return _asr_http(wav)
        print("[voice] STT 服务不可用, 回退 transformers GPU 后端")
    except Exception as e:
        print(f"[voice] HTTP STT 异常({e}), 回退 transformers GPU 后端")
    return _asr_qwen3(wav)


# 闲聊情绪词典: 关键词 -> 表情包情绪(用于闲聊互动, 未命中默认 可爱卖萌)
CHAT_EMOTIONS = [
    (r"厉害|棒|强|牛|可以啊|不错|好样的", "点赞加油"),
    (r"谢谢|感谢|多谢|爱你", "比心感谢"),
    (r"可爱|萌|乖|乖乖", "可爱卖萌"),
    (r"哈哈|开心|高兴|好玩|有趣|嘻嘻", "开心对视"),
    (r"难过|想哭|伤心|呜呜|呜咽", "大哭流泪"),
    (r"生气|气死|烦死|可恶|混蛋", "生气发怒"),
    (r"困|累|睡|觉|打哈欠", "已然睡着"),
    (r"尴尬|社死|无语死了|汗", "流汗尴尬"),
    (r"为什么|啥意思|不懂|不明白|怎么会", "疑惑不解"),
    (r"得瑟|骄傲|自豪|我最", "自信得意"),
    (r"坏|嘿嘿|坏笑|使坏", "坏笑"),
    (r"礼物|奖励|送我|拿到", "得到礼物"),
    (r"叹气|唉|无奈|没办法", "无奈叹气"),
]
DEFAULT_CHAT_EMOTION = "可爱卖萌"

TRAD2SIMP = str.maketrans("來給們聽聽見個沒說話樣後時,种里见过为发张问题对当", "来给们听听见个没说话样后时,种里见过为发张问题对当")
# 意图词高频繁体补充(whisper常出繁体); 字典对构造, 免maketrans等长限制
TRAD2SIMP.update(str.maketrans({
    "傳": "传", "飛": "飞", "動": "动", "帶": "带", "尋": "寻", "欄": "栏",
    "調": "调", "變": "变", "龜": "龟", "點": "点", "換": "换", "劇": "剧",
    "講": "讲", "詞": "词", "貼": "贴", "紙": "纸", "臺": "台", "達": "达",
    "導": "导", "語": "语", "訊": "讯", "無": "无", "開": "开", "關": "关",
    "體": "体", "歷": "历", "聲": "声", "幫": "帮", "蘋": "苹", "檔": "档", "錶": "表", "後": "后", "裡": "里", "麼": "么"}))

PAIRS_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "voice_pairs.jsonl")
STATE_FILE = os.path.join(os.environ.get("APPDATA") or os.path.expanduser("~/AppData/Roaming"), "Godot", "app_userdata",
                          "StardewValley", "state", "world_state.json")

# ===== 多项目路由(v6.12: 太空游戏 gdgs_lab 复用同一 STT/意图/命令管线)=====
PROJECT = "StardewValley"   # --project 切换; voice/inbox/events 目录随项目走

def switch_project(name: str) -> None:
    global PROJECT, VOICE_DIR, INBOX, EVENTS, STATE_FILE
    PROJECT = name
    _base = os.path.join(os.environ.get("APPDATA") or os.path.expanduser("~/AppData/Roaming"),
                         "Godot", "app_userdata", name)
    VOICE_DIR = os.path.join(_base, "voice")
    INBOX = os.path.join(_base, "commands", "inbox.jsonl")
    EVENTS = os.path.join(_base, "logs", "game_events.jsonl")
    STATE_FILE = os.path.join(_base, "state", "world_state.json")

# 太空游戏意图词表: 轨迹圈(轨道线)显示/隐藏/切换
SPACE_ORBIT_WORDS = r"轨迹圈|轨道圈|轨道线|轨道|轨迹"

def classify_space(text: str) -> list:
    """gdgs_lab 意图表: 命中返回命令列表, 未命中返回 []。"""
    text = text.translate(TRAD2SIMP)
    eva = classify_eva(text)          # EVA
    if eva:
        return eva   # EVA 命中即短路(播报/查询/询问/静音)
    cmds = []
    def submit(cmd_type, params, tag):
        cmds.append({"command_id": f"voice_{int(time.time()*1000)}_{tag}",
                     "command_type": cmd_type, "actor": "voice", "params": params})
    # v6.16 传送意图: "传送到木星/去火星/前往地球…" -> teleport 命令(即时改玩家位置)
    PLANET_CN = [("天王星", "uranus"), ("海王星", "neptune"), ("水星", "mercury"),
                 ("金星", "venus"), ("地球", "earth"), ("火星", "mars"),
                 ("木星", "jupiter"), ("土星", "saturn")]
    if re.search(r"传送|跳转|跃迁|前往|移动到|到.*去", text):
        for cn, en in PLANET_CN:
            if cn in text:
                submit("teleport", {"target": en, "cn": cn}, "tp_" + en)
                break
    if re.search(SPACE_ORBIT_WORDS, text):
        if re.search(r"切换|开关", text):
            submit("toggle_orbit_lines", {}, "orbit_toggle")
        elif re.search(r"隐藏|关掉|关闭|去掉|藏起来|收起|消失|不显示|别显示", text):
            submit("set_orbit_lines", {"visible": False}, "orbit_off")
        else:
            submit("set_orbit_lines", {"visible": True}, "orbit_on")
    if not cmds:
        cmds = classify_eva_bare(text)   # 免唤醒裸查询: "查询/任务/战况"直接路由 EVA 播报
    return cmds
ASR_BACKEND = "qwen3asr-gguf-http"   # 主路: llama.cpp Q4_K_M 1.7B@8124(零显存); 备胎: transformers GPU(whisper/sensevoice 已卸载)

EMO_EVENT_SOURCE = {1.0: "dict", 0.0: "default"}   # 其余 score -> semantic
_emo_seq = 0


def log_emotion_event(text: str, emo: str, score: float, wav: str) -> None:
    """情绪打分落游戏日志: game_events.jsonl 追加 chat.emotion 事件(带默认值兜底)。
    情绪聚合/自适应选曲层从这里取数, 不用再从贴纸文件名反推。"""
    rec = {"event_id": f"chat.emotion.{int(time.time()*1000)}_{_emo_seq}",
           "event_type": "chat.emotion",
           "frame": 0, "time": time.strftime("%Y-%m-%dT%H:%M:%S"),
           "unix_time": time.time(),
           "data": {"text": text, "emotion": emo, "score": score,
                    "source": EMO_EVENT_SOURCE.get(score, "semantic"),
                    "wav": os.path.basename(wav)}}
    try:
        with open(EVENTS, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    except OSError as e:
        print(f"[voice] chat.emotion写入失败: {e}")


def log_pair(text: str, action: str, sticker: str = None) -> None:
    """收集训练对: (转写文本, 最终动作, 表情) -> JSONL, 供后续微调意图/情绪路由器。"""
    import json as _json
    rec = {"ts": time.time(), "text": text,
           "text_simp": text.translate(TRAD2SIMP),
           "action": action, "sticker": sticker}
    with open(PAIRS_FILE, "a", encoding="utf-8") as f:
        f.write(_json.dumps(rec, ensure_ascii=False) + "\n")


def _extract_place(text: str):
    """'传送去村口' -> 'village'(PLACE_WORDS 命中即返回 key)"""
    for pat, key in PLACE_WORDS:
        if re.search(pat, text):
            return key
    return None


def _extract_item(text: str):
    """'背包里的木头' -> ('WOOD', '木头')"""
    for pat, key in ITEM_WORDS:
        m = re.search(pat, text)
        if m:
            return key, m.group(0)
    return None


def _extract_coords(text: str):
    """'坐标 63 2'/'坐标1016,40' -> (x, y)。两轴都<=200 视为瓦片坐标(*16+8 取中心)。"""
    m = re.search(r"坐标[:：]?\s*(\d{1,4})\s*[,，\s]\s*(\d{1,4})", text)
    if not m:
        return None
    x, y = float(m.group(1)), float(m.group(2))
    if x <= 200 and y <= 200:
        x, y = x * 16 + 8, y * 16 + 8
    return x, y


def _extract_npc(text: str):
    """'去找王桂兰那里' -> '王桂兰'(村民名2~3汉字; 兼容 Cat 等英文节点名)"""
    m = re.search(r"(?:去找|找找|找一找|找一?[个下]|带我去|去见|"
                  r"走到|走去|走向|前往|寻路到?|找)[:：]?\s*"
                  r"([一-龥]{2,3})(?:那里|那儿|身边|旁边)?", text)
    if m:
        return m.group(1)
    m = re.search(r"(?:去找|带我去|去见|找)[:：]?\s*([A-Za-z]{3,8})(?:那里)?", text)
    return m.group(1) if m else None


def world_state(max_age: float = 8.0):
    """读游戏端每2s导出的世界状态; 过期/缺失返回 None(状态不可知)"""
    try:
        if time.time() - os.stat(STATE_FILE).st_mtime > max_age:
            return None
        with open(STATE_FILE, encoding="utf-8") as f:
            return json.load(f)
    except (OSError, ValueError):
        return None


def inventory_search(zh_name: str):
    """前置参数搜索: 在背包里找物品。先枚举物品(词表反查), 再模糊匹配AI动态物品。
    命中返回参数dict, 没找到/状态不可知返回 None, 由上层走统一fallback。"""
    st = world_state()
    if not st or not zh_name:
        return None
    inv = st.get("inventory", {})
    item = _extract_item(zh_name)
    if item and int(inv.get(item[0], 0)) > 0:
        return {"item": item[0], "name": item[1], "count": int(inv[item[0]])}
    for e in st.get("custom_items", []):
        if (zh_name in str(e.get("name", "")) or zh_name in str(e.get("id", ""))) and int(e.get("count", 0)) > 0:
            return {"custom": e["id"], "name": str(e.get("name", e["id"])), "count": int(e["count"])}
    return None



def _notify(msg: str) -> None:
    """gdgs_lab: 转写反馈写进 inbox, 游戏 HUD 经 notify 接口播报。"""
    cmd = {"command_id": f"voice_{int(time.time()*1000)}_notify",
           "command_type": "notify", "actor": "voice", "params": {"text": msg}}
    try:
        os.makedirs(os.path.dirname(INBOX), exist_ok=True)
        with open(INBOX, "a", encoding="utf-8") as f:
            f.write(json.dumps(cmd, ensure_ascii=False) + "\n")
        print(f"[voice] notify: {msg}")
    except OSError as e:
        print(f"[voice] notify 写入失败: {e}")

def classify(text: str) -> list[dict]:
    """文本 -> 命令列表(intent规则)。未命中返回 []。"""
    global _last_track
    text = text.translate(TRAD2SIMP)
    if PROJECT != "StardewValley":
        return classify_space(text)
    cmds = []
    def submit(cmd_type, params, tag):
        cmds.append({"command_id": f"voice_{int(time.time()*1000)}_{tag}",
                     "command_type": cmd_type, "actor": "voice",
                     "params": params})
    if re.search(r"下雨|雨天|暴雨", text):
        # 背景音与音乐并行: 雨声走 Ambient 常驻循环, 不占歌单、不打断 BGM
        import sticker_finder; sticker_finder.show(WEATHER_STICKER["rain"], seconds=1.5)
        submit("play_ambient", {"name": "rain"}, "amb_rain")
    elif re.search(r"晴天|白天|放晴|天晴", text):
        # 晴天只关环境音, 音乐照常放
        import sticker_finder; sticker_finder.show(WEATHER_STICKER["day"], seconds=1.5)
        submit("play_ambient", {"name": "off"}, "amb_off")
    elif re.search(r"生成.*(歌|曲|音乐)|来(一|点)首新|新.*(?:曲子|音乐)|(钢琴|小提琴|吉他|管弦).*(?:曲|音乐)", text):
        # 投递音乐生成任务给单例服务 music_gen_service.py(串行消费, 防并发拉模型)
        import music_gen_service as _mgs
        job = _mgs.text_to_job(text) or {"slot": "cozy", "period": "Romantic",
                                         "composer": "Schumann", "instr": "Piano"}
        os.makedirs(_mgs.JOBS_DIR, exist_ok=True)
        jf = os.path.join(_mgs.JOBS_DIR, f"job_{int(time.time()*1000)}.json")
        with open(jf, "w", encoding="utf-8") as f:
            json.dump(job, f, ensure_ascii=False)
        print(f"[voice] 音乐生成任务已投递: {job}")
        cmds.append({"command_id": "_handled", "command_type": "_music_generating",
                     "actor": "voice", "params": {}})   # 哨兵: 已处理
    # whisper 常把"切歌"误听成"切割/切个/七哥/柒哥/且歌/砌歌", 全部归到切歌意图
    elif re.search(r"换一?(?:[首个张]|首[歌曲]|个[歌曲]|歌|曲|音乐|bgm)|切(?:一?[下首两歌]|歌|割|[個个])|下一[首曲个]|换个(?:音乐|曲子|bgm)|(?:七|柒)哥|且歌|砌歌", text) and not re.search(r"切[个一](?!歌|曲|音乐|bgm|BGM|首)[一-龥]", text):
        # 真相在游戏侧: 读最后一次 music.switched, 在歌单里顺位下一首(幂等)
        cur = _current_track_from_log()
        other = MUSIC_PLAYLIST[(MUSIC_PLAYLIST.index(cur) + 1) % len(MUSIC_PLAYLIST)] \
            if cur in MUSIC_PLAYLIST else "day"
        _last_track = other
        # 切歌进行中提示: 同步弹指定表情包告知正在切(无需确认, 单向提示)
        import sticker_finder
        sticker_finder.show(SWITCHING_STICKER, seconds=1.5)
        submit("play_music", {"track": other}, "switch")
    elif re.search(r"剧情|讲故事|说书|新包|换个包|来一[个秒]|再来一", text):
        # 转交剧情包加载器(最新一个包)
        packs = sorted(glob.glob(os.path.join(os.path.dirname(
            os.path.abspath(__file__)), "packs", "pack_*.json")))
        if packs:
            subprocess.Popen([sys.executable, "hot_load_pack.py", packs[-1]],
                             cwd=os.path.dirname(os.path.abspath(__file__)))
            print(f"[voice] 触发剧情包: {os.path.basename(packs[-1])}")
        cmds.append({"command_id": "_handled", "command_type": "_pack_loading",
                     "actor": "voice", "params": {}})   # 哨兵: 已处理, 勿走闲聊兜底
    elif re.search(r"表情|贴纸|表情包", text):
        m = re.search(r"表情(?:包)?|贴纸", text)
        kw = text[m.end():].strip(" ：:，,的")
        if not kw:
            kw = text[:m.start()].strip("给我来个张看个")
        if kw:
            import sticker_finder
            path = sticker_finder.show(kw) or sticker_finder.show("可爱卖萌")
            print(f"[voice] 表情 {kw!r} -> {path if path else '未找到'}")
            cmds.append({"command_id": "_handled", "command_type": "_sticker_shown",
                         "actor": "voice", "params": {}})   # 哨兵: 标记意图已处理
    elif SPEED_GATE.search(text) and not text.startswith("台词"):
        # 移速热调: 数字倍率 > 词表 > 快/慢兜底; whisper常把"移速"听成"一速"、"调"发繁体"調"
        mult = None
        for pat, v in SPEED_WORDS:
            if re.search(pat, text):
                mult = v
                break
        if mult is None:
            mult = 1.6 if re.search(r"提速|加快|加速|快", text) else (0.6 if re.search(r"减速|放慢|慢", text) else 1.0)
        submit("set_speed", {"multiplier": mult}, "speed")
    elif re.search(r"走去|走到|走向|前往|带我去|寻路|去见|去找|找找|找一找|找一?[个下]|找", text)             and not text.startswith("台词"):
        # 寻路: 坐标 > 物品(转背包查询) > 地名 > NPC(实时位置停身旁); 都没给 -> 村中心
        coords = _extract_coords(text)
        item = _extract_item(text)
        npc = _extract_npc(text)
        place = _extract_place(text)
        m_obj = re.search(r"找一?[个下][:：]?\s*([一-龥A-Za-z]{2,6})", text)
        if coords:
            submit("walk_to", {"x": coords[0], "y": coords[1]}, "walk")
        elif item:
            # "找木头/找一下苹果" 找的是物品: 转背包查询
            submit("find_item", {"item": item[0], "name": item[1]}, "find")
        elif m_obj:
            # "找个猫铃铛" 自定义物品(AI动态创建), 交给游戏端按名字搜
            submit("find_item", {"name": m_obj.group(1)}, "find")
        elif place:
            # "带我去村中心广场": 地名优先于人名, 防止地名被误当NPC
            submit("walk_to", {"place": place}, "walk")
        elif npc:
            submit("walk_to", {"npc": npc}, "walk")
        else:
            submit("walk_to", {"place": "village"}, "walk")
    elif re.search(r"传送|瞬移|飞到|挪到|移动到|拉到", text) and not text.startswith("台词"):
        # 传送: 坐标 > 地名; 都没给 -> 默认村中心
        coords = _extract_coords(text)
        place = _extract_place(text)
        if coords:
            submit("tp", {"x": coords[0], "y": coords[1]}, "tp")
        elif place:
            submit("tp", {"place": place}, "tp")
        else:
            submit("tp", {"place": "village"}, "tp")
    elif re.search(r"背包|物品栏|包裹|口袋", text) and not text.startswith("台词"):
        # 背包总览/查询: 带物品词查单个, 只说背包播总览
        item = _extract_item(text)
        if item:
            submit("find_item", {"item": item[0], "name": item[1]}, "find")
        else:
            submit("find_item", {"list": True}, "find")
    elif _extract_item(text) and re.search(r"有(?:没)?有|多少|几个|查一?下|看看|翻翻", text) and not text.startswith("台词"):
        # 不带"背包"二字的物品查询: "还有没有木头"/"我有几个玉米"
        item = _extract_item(text)
        submit("find_item", {"item": item[0], "name": item[1]}, "find")
    elif re.search(r"切[出开掉]|切片", text) and not text.startswith("台词"):
        # 标杆三段式: 门控 -> 前置搜索(背包) -> 动作 / fallback(表情+播报)
        item = _extract_item(text)
        m_obj = re.search(r"切[出开掉一][个只块]?:?\s*([一-龥A-Za-z]{2,6})", text)
        zh = item[1] if item else (m_obj.group(1) if m_obj else "")
        hit = inventory_search(zh)
        if hit:
            submit("cut_item", hit, "cut")
        else:
            # 统一fallback: 没找到 -> 疑惑表情 + 游戏内播报"背包里没有X"
            import sticker_finder
            sticker_finder.show("疑惑不解", seconds=1.5)
            if zh:
                submit("find_item", {"name": zh}, "find")
            cmds.append({"command_id": "_handled", "command_type": "_cut_fallback",
                         "actor": "voice", "params": {}})
    else:
        m = re.search(r"台词[:：]?\s*(.+)", text)
        lines = [l.strip() for l in re.split(r"[,，。;；]", m.group(1)) if l.strip()] if m else []
        if not lines and len(text) <= 30 and re.search(r"喵|猫", text):
            lines = [text.strip()]
        if lines:
            submit("set_dialog", {"npc": "Cat", "lines": lines[:5]}, "dialog")
    return cmds


def _seed_seen():
    """幂等: 启动时把 voice 目录里已有的 wav 全标记已处理, 防止重启重放历史音频。"""
    for wav in glob.glob(os.path.join(VOICE_DIR, "input_*.wav")):
        _seen.add(wav)
    print(f"[voice-intent] 跳过历史音频 {len(_seen)} 条(防重放)")


def _current_track_from_log() -> str:
    """幂等: 从 game_events.jsonl 读最后一次 music.switched, 单一事实源。"""
    try:
        with open(EVENTS, encoding="utf-8") as f:
            last = None
            for line in f:
                if '"music.switched"' in line:
                    last = line
            if last:
                return json.loads(last)["data"].get("track", "day")
    except (OSError, json.JSONDecodeError):
        pass
    return _last_track


def main():
    import argparse
    ap = argparse.ArgumentParser(description="语音意图服务(多项目)")
    ap.add_argument("--project", default="StardewValley",
                    help="Godot app_userdata 项目名: StardewValley | gdgs_lab")
    args = ap.parse_args()
    switch_project(args.project)

    # 单实例锁: 端口占用即已有服务在跑, 直接退出
    import socket
    lock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    _lock_port = 52771 if PROJECT == "StardewValley" else 52771 + (hash(PROJECT) % 97)
    try:
        lock.bind(("127.0.0.1", _lock_port))
    except OSError:
        print(f"[voice-intent] instance already running (port {_lock_port}), exit")
        return
    os.makedirs(VOICE_DIR, exist_ok=True)
    _seed_seen()
    print(f"[voice-intent] watching {VOICE_DIR}")
    while True:
        for wav in glob.glob(os.path.join(VOICE_DIR, "input_*.wav")):
            if wav in _seen:
                continue
            _seen.add(wav)
            try:
                t0 = time.time()
                text = transcribe(wav)
                print(f"[voice] {os.path.basename(wav)} ({time.time()-t0:.1f}s): {text!r}")
                if PROJECT != "StardewValley":
                    if not text:
                        _notify("没听清(静音或太短), 请再试一次")
                        continue
                    cmds = classify(text)   # classify 内部已按 PROJECT 分流到 classify_space
                    if not cmds:
                        _notify(f"未识别指令: {text[:16]}")
                        continue
                    with open(INBOX, "a", encoding="utf-8") as f:
                        for c in cmds:
                            f.write(json.dumps(c, ensure_ascii=False) + "\n")
                            print(f"[voice] -> {c['command_type']} {str(c['params'])[:80]}")
                    continue
                # 转写即打分: 无论后续走指令还是闲聊, 情绪分数先落日志(空文本=default)
                from chat_emotion_router import route_emotion
                _emo, _sc = route_emotion(text) if text else (DEFAULT_CHAT_EMOTION, 0.0)
                log_emotion_event(text, _emo, _sc, wav)
                if not text:
                    import sticker_finder
                    print("[voice] 空转写 -> 疑惑不解:", sticker_finder.show("疑惑不解"))
                    continue
                cmds = classify(text)
                if not cmds:
                    # 闲聊也互动: 快速词典 + qwen3-embedding 语义情绪路由(chat_emotion_router)
                    import re as _re, sticker_finder
                    emo, score = _emo, _sc   # 复用转写时已算好的打分, 不重算
                    path = sticker_finder.show(emo, seconds=2.2)
                    print(f"[voice] 闲聊 -> {emo}({score}) -> {path}")
                    simp = text.translate(TRAD2SIMP)
                    log_pair(text, f"chat:{emo}", path)
                    # 疑似指令但未识别仍然弹疑惑
                    if _re.search(r"^[来給给换打播放设置关闭下上]?[一这那]|请|帮我|给我|打开|切换", simp):
                        print("[voice] 疑似指令未识别 -> 疑惑不解:",
                              sticker_finder.show("疑惑不解"))
                    continue
                log_pair(text, cmds[0]["command_type"])
                with open(INBOX, "a", encoding="utf-8") as f:
                    for c in cmds:
                        if c["command_type"].startswith("_"):
                            continue   # 哨兵命令不进inbox, 只标记意图已处理
                        f.write(json.dumps(c, ensure_ascii=False) + "\n")
                        print(f"[voice] -> {c['command_type']} {str(c['params'])[:80]}")
            except Exception as e:
                print(f"[voice] error {wav}: {e}")
        time.sleep(0.8)


if __name__ == "__main__":
    main()
