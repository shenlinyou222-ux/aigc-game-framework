# -*- coding: utf-8 -*-
"""EVA 舰载语音播报意图(gdgs_lab 太空游戏专用)。

v0.9.10: 游戏 _feed 文字下线(出戏)后由语音播报接管; 词库/静态话术在
eva_broadcasts.json(单一事实源, 供 TTS 预合成)。命令由游戏端 CMDSRV 执行:
  eva_query  {topic: ack|mission|status|resources|help|unknown, heard?}
  eva_repeat {}                      -> 重播上一条播报
  eva_mute   {on: bool}              -> 静音/恢复
唤醒: 话语以 EVA 开头(Qwen3-ASR 常出 EVA/伊娃/诶哇/艾娃等变体)。
"""
import re
import time

EVA_WAKE = re.compile(r"^\s*(?:EVA|伊娃|诶哇|衣娃|艾娃)[，,。.、:：!！~～\s]*", re.IGNORECASE)
EVA_TOPICS = [
    (r"(?:当前|现在)?任务|使命|目标|现在做什[么麽]|干什[么麽]", "mission"),
    (r"状态|战况|战报|情况|汇报|报告|局势|损伤", "status"),
    (r"资源|物资|经济|补给", "resources"),
    (r"帮助|能做什[么麽]|会什[么麽]|指令|命令|怎么用|能干嘛", "help"),
]
EVA_REPEAT_WORDS = r"再说一?遍|重复|刚[才刚]?说[的了]|没听清|什么意思"
EVA_MUTE_WORDS = r"安静|闭嘴|别说话?了|别播了|停止播报|静音"
EVA_UNMUTE_WORDS = r"恢复播报|继续说|取消静音|可以说话|开声音"


def classify_eva(text: str) -> list:
    """EVA 唤醒播报: 以唤醒词开头才命中返回播报命令; 否则返回 [](不影响其他意图)。"""
    m = EVA_WAKE.match(text)
    if not m:
        return []
    rest = text[m.end():].strip()
    cmds = []

    def submit(cmd_type, params, tag):
        cmds.append({"command_id": "voice_%d_%s" % (int(time.time() * 1000), tag),
                     "command_type": cmd_type, "actor": "voice", "params": params})

    if not rest:                                   # 只喊名字 -> 应答+用法提示
        submit("eva_query", {"topic": "ack"}, "eva_ack")
    elif re.search(EVA_MUTE_WORDS, rest):
        submit("eva_mute", {"on": True}, "eva_mute")
    elif re.search(EVA_UNMUTE_WORDS, rest):
        submit("eva_mute", {"on": False}, "eva_unmute")
    elif re.search(EVA_REPEAT_WORDS, rest):
        submit("eva_repeat", {}, "eva_repeat")
    else:
        for pat, topic in EVA_TOPICS:
            if re.search(pat, rest):
                submit("eva_query", {"topic": topic}, "eva_" + topic)
                break
        else:                                      # 唤醒但意图不明 -> 播"询问"话术(意图识别询问)
            submit("eva_query", {"topic": "unknown", "heard": rest[:24]}, "eva_unknown")
    return cmds

# ===== 免唤醒裸查询(v?: 太空项目语音助手即 EVA, 明显查询意图免喊唤醒词) =====
# 词表顺序即优先级: 具体主题在前, 泛化词(查询/任务)兜底归 mission。
# 控制类(安静/恢复)不进裸词表 —— 避免闲聊误触静音。
EVA_BARE_WORDS = [
    (r"再说一?遍|重复一遍?", "repeat"),
    (r"帮助|能做什[么麽]|怎么用", "help"),
    (r"资源|物资|经济|补给", "resources"),
    (r"战况|战报|汇报|报告|状态|局势", "status"),
    (r"(?:查询|查一下|查查|问下?|播报)(?:一下)?|任务|目标|使命", "mission"),
]


def classify_eva_bare(text: str) -> list:
    """无唤醒词时的裸查询兜底: 命中查询类词即路由 EVA 播报(如裸"查询"->mission)。"""
    for pat, topic in EVA_BARE_WORDS:
        if re.search(pat, text):
            return [{"command_id": "voice_%d_ebabare_%s" % (int(time.time() * 1000), topic),
                     "command_type": "eva_repeat" if topic == "repeat" else "eva_query",
                     "actor": "voice",
                     "params": {} if topic == "repeat" else {"topic": topic}}]
    return []
