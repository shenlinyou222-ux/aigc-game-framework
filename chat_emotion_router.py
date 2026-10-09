# -*- coding: utf-8 -*-
"""闲聊情绪路由(语义层): 复用 aigc_core.OllamaLogEmbedder (qwen3-embedding:0.6b)。

用法:
    from chat_emotion_router import route_emotion
    emo, score = route_emotion("你會哭嗎")   # -> ("大哭流泪", 0.62)

设计:
- 每个情绪一组示例句, 首次调用时批量 embed 并缓存(进程内 + json 缓存文件);
- 输入文本 embed 后与各情绪质心做余弦相似度, 取最高;
- score < MIN_SCORE 时回退 DEFAULT(可爱卖萌), 上层可再走词典/兜底;
- 独立进程也可直接命令行测试: python chat_emotion_router.py "文本"
"""
import json
import math
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

MODEL = "qwen3-embedding:0.6b"
CACHE_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                          "chat_emotion_centroids.json")
MIN_SCORE = 0.35          # 余弦阈值: 低于则视为不确定
DEFAULT_EMOTION = "可爱卖萌"

# 情绪 -> 示例句(覆盖繁体/口语/否定问句; 质心比单句更稳)
EMOTION_EXEMPLARS = {
    "点赞加油": ["你真厉害", "做得好棒啊", "太强了", "干得漂亮", "加油你可以的"],
    "比心感谢": ["谢谢你啦", "多谢多谢", "太感谢了", "爱你哟", "辛苦了谢谢你"],
    "可爱卖萌": ["你好可爱呀", "萌死了", "乖乖的", "小可爱", "好乖哦"],
    "开心对视": ["哈哈真好玩", "我好开心", "太高兴了", "有趣有趣", "嘻嘻笑死我了"],
    "大哭流泪": ["我好难过", "想哭呜呜", "你會哭嗎", "不会伤心吗", "好想大哭一场", "眼泪要掉下来了"],
    "生气发怒": ["气死我了", "真讨厌烦死了", "可恶啊", "太让人生气了", "混蛋"],
    "已然睡着": ["好困想睡觉", "困死了", "打个哈欠", "困得睁不开眼", "该睡觉了"],
    "流汗尴尬": ["好尴尬啊", "社死了", "无语死了", "出了一身汗", "这也太尴尬了吧"],
    "疑惑不解": ["为什么呢", "啥意思啊", "我不明白", "怎么会这样", "这是什么东西"],
    "自信得意": ["我最厉害", "得瑟一下", "我很自豪", "看我的厉害", "我可是最强的"],
    "坏笑": ["嘿嘿坏笑", "我要使坏了", "坏坏的主意", "嘿嘿嘿", "偷偷搞点坏事"],
    "得到礼物": ["收到礼物了好开心", "这是给我的奖励吗", "送我一个小礼物吧", "拿到奖品了", "有礼物拿诶"],
    "无奈叹气": ["唉没办法", "叹口气", "很无奈啊", "只能这样了", "唉算了吧"],
}

# 快速词典路径(高置信关键词优先, 命中即返回, 省 embed 调用)
FAST_RULES = [
    (r"谢谢|感谢|多谢|爱你", "比心感谢"),
    (r"厉害|棒|好样的|干得漂亮", "点赞加油"),
    (r"哈哈|嘻嘻|开心|高兴|好玩|有趣", "开心对视"),
    (r"哭|伤心|难过|呜呜", "大哭流泪"),
    (r"气死|生气|烦死|可恶|混蛋", "生气发怒"),
    (r"困|累|睡|哈欠", "已然睡着"),
    (r"尴尬|社死|无语死", "流汗尴尬"),
    (r"为什么|啥意思|不懂|不明白|怎么会", "疑惑不解"),
    (r"礼物|奖励|送我", "得到礼物"),
    (r"叹气|唉|无奈", "无奈叹气"),
    (r"可爱|萌|乖", "可爱卖萌"),
]

# 否定守卫: 关键词前紧邻 别/不/没/莫/勿 时跳过词典, 交给语义层判
# 注意: lookbehind 必须逐分支加, 拼在整个交替式开头只保护第一个分支
_NEG_GUARD = r"(?<![别不没莫勿])"
_FAST_RULES_C = [
    ("(" + "|".join(_NEG_GUARD + alt for alt in p.split("|")) + ")", e)
    for p, e in FAST_RULES
]

TRAD2SIMP = str.maketrans(
    "來給們聽聽見個沒說話樣後時,种里见过为发张问题对当會傷心開心氣", 
    "来给们听听见个没说话样后时,种里见过为发张问题对当会伤心开心气")

_embedder = None
_centroids = None   # {emotion: [float,...]}


def _get_embedder():
    global _embedder
    if _embedder is None:
        from aigc_core.embedders import OllamaLogEmbedder
        _embedder = OllamaLogEmbedder(model=MODEL)
    return _embedder


def _cos(a, b):
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a)) or 1.0
    nb = math.sqrt(sum(x * x for x in b)) or 1.0
    return dot / (na * nb)


def _build_centroids():
    """示例句 embed -> 每情绪取质心(归一化)。带文件缓存避免每次重启重算。"""
    global _centroids
    if _centroids is not None:
        return _centroids
    emb = _get_embedder()
    if os.path.exists(CACHE_FILE):
        try:
            with open(CACHE_FILE, encoding="utf-8") as f:
                _centroids = json.load(f)
            return _centroids
        except Exception:
            pass
    cents = {}
    for emo, sents in EMOTION_EXEMPLARS.items():
        vecs = [emb.embed_text(s) for s in sents]
        n = len(vecs)
        dim = len(vecs[0])
        c = [sum(v[i] for v in vecs) / n for i in range(dim)]
        norm = math.sqrt(sum(x * x for x in c)) or 1.0
        cents[emo] = [x / norm for x in c]
    _centroids = cents
    try:
        with open(CACHE_FILE, "w", encoding="utf-8") as f:
            json.dump(cents, f)
    except Exception:
        pass
    return _centroids


# 否定词表抄自 cnsenti(hownet/deny.pkl) + dutir 7类情绪词典
# 快速词典保留, 但否定词紧邻关键词时跳过(奇偶计数: 奇数否定=反转)
_DICT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "emotion_dict")

def _load_pkl_keys(name):
    import pickle
    with open(os.path.join(_DICT_DIR, name), "rb") as f:
        d = pickle.load(f)
    return d.keys() if isinstance(d, dict) else d

DENY_WORDS = set(_load_pkl_keys("deny.pkl"))   # 不/不是/别/没有/莫/未/勿/甭...
DUTIR_MAP = {   # dutir 情绪类 -> 我们的贴纸情绪
    "乐": "开心对视", "好": "点赞加油", "哀": "大哭流泪", "怒": "生气发怒",
    "惧": "疑惑不解", "恶": "生气发怒", "惊": "疑惑不解",
}
_DUTIR_WORDS = {}   # word -> dutir类别(仅高频常用词, 控制规模)


def _init_dutir():
    """只保留长度>=2且常见的词, 避免'金字塔'这类误命中; 惰性加载一次。"""
    global _DUTIR_WORDS
    if _DUTIR_WORDS:
        return
    for cat in DUTIR_MAP:
        try:
            for w in _load_pkl_keys(f"{cat}.pkl"):
                if isinstance(w, str) and 2 <= len(w) <= 4:
                    _DUTIR_WORDS[w] = cat
        except FileNotFoundError:
            pass


def _lexicon_route(simp):
    """抄 cnsenti sentiment.py 的否定反转逻辑(简化): 长词优先匹配情绪词,
    紧邻前方奇数个否定词 -> 跳过该词; 命中即返回我们的情绪。"""
    _init_dutir()
    # 合并查: 我们的词典 + dutir, 长词优先
    for i in range(len(simp) - 1):
        for L in (4, 3, 2):
            w = simp[i:i + L]
            if w in _DUTIR_WORDS:
                # 前3字窗口内出现任意否定词(不/别/没/莫/未/勿/甭...) -> 反转, 弃用
                if any(dw in simp[max(0, i - 3):i] for dw in DENY_WORDS):
                    continue
                return DUTIR_MAP[_DUTIR_WORDS[w]]
    return None


def route_emotion(text: str):
    """文本 -> (情绪, 相似度)。词典+否定守卫优先, 语义兜底, 低于阈值回默认。"""
    simp = text.translate(TRAD2SIMP).strip()
    if not simp:
        return DEFAULT_EMOTION, 0.0
    for pat, emo in _FAST_RULES_C:      # 1) 快速词典(带否定守卫)
        if re.search(pat, simp):
            return emo, 1.0
    lex = _lexicon_route(simp)          # 2) dutir词典+否定反转
    if lex:
        return lex, 1.0
    try:                                  # 2) 语义最近邻
        cents = _build_centroids()
        v = _get_embedder().embed_text(simp)
        best, best_s = DEFAULT_EMOTION, -1.0
        for emo, c in cents.items():
            s = _cos(v, c)
            if s > best_s:
                best, best_s = emo, s
        if best_s >= MIN_SCORE:
            return best, round(best_s, 3)
        return DEFAULT_EMOTION, round(best_s, 3)
    except Exception as e:                # ollama 挂了 -> 默认, 不阻塞语音链路
        print(f"[chat-emo] embed失败({e}), 用默认情绪")
        return DEFAULT_EMOTION, 0.0


if __name__ == "__main__":
    for t in sys.argv[1:]:
        print(f"{t!r} -> {route_emotion(t)}")
