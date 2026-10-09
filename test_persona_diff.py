# -*- coding: utf-8 -*-
"""人设差异化验证: 相同场景/事实/节拍, 只换 characters 卡 → 输出必须差异化"""
import json, urllib.request, re, yaml

OLLAMA = "http://localhost:11434/api/chat"

def gen(system, history_text):
    msgs = [{"role": "system", "content": system},
            {"role": "user", "content":
             (f"已有对话:\n{history_text}\n\n" if history_text else "") +
             "只输出该角色说出口的一句台词，25字内，可含(动作神态)。禁止emoji/英文/景物描写。"}]
    body = json.dumps({"model": "qwen3.5:0.8b", "messages": msgs, "stream": False,
                       "think": False, "options": {"num_predict": 80,
                       "temperature": 0.7, "top_p": 0.9, "presence_penalty": 0.3}}).encode("utf-8")
    req = urllib.request.Request(OLLAMA, data=body, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=120) as r:
        d = json.loads(r.read())
    line = (d["message"].get("content") or "").strip().splitlines()[0]
    line = re.split(r"[|｜]", line)[-1].strip()[:60]
    return re.sub(r"[\U0001F300-\U0001FAFF\u2700-\u27BF\uFE0F]", "", line).strip("\"「'")

# 相同节拍: 只与"暴风雨毁鸡舍+捡到锄头"有关, 与人设无关
BEATS = [
    ("愤怒地提到鸡舍被暴风雨掀了、死了3只鸡"),
    ("对方安慰时，提到收获节庆典推迟到下周"),
    ("掏出在雷劈树根下捡到的星陨锄头，得意展示"),
    ("听到'明天帮你处理'之类的提议后的反应"),
]

# 三套人设卡: 同一场景同一处境, 只有性格/身份/口癖不同
PERSONAS = {
    "A·倔强农女(艾拉)": """
身份: 22岁农场少女, 家里的鸡舍就是她家的命根子
性格: 倔强、嘴硬心软、讨厌被同情
语言风格: 口头禅'哼', 说话冲但藏着感激, 每句不超25字
当前情绪底色: 又气又心疼""",
    "B·温柔大小姐(薇拉)": """
身份: 20岁富商之女, 来村里避暑, 对农家事好奇但不懂
性格: 温柔、共情力强、有点娇气, 习惯用敬语
语言风格: 轻声细语爱用'呢/呀', 每句不超25字
当前情绪底色: 同情又不知所措""",
    "C·糙汉老猎户(铁牛)": """
身份: 45岁老猎户, 艾拉的邻居, 看着艾拉长大
性格: 豪爽、大大咧咧、把关心藏在糙话里
语言风格: 嗓门大爱说'娘的/咋整', 每句不超25字
当前情绪底色: 心疼但嘴上不饶人""",
}

results = {}
for pname, card in PERSONAS.items():
    history, lines = [], []
    for i, beat in enumerate(BEATS):
        system = f"你扮演的角色设定:\n{card}\n场景: 收获节前夜, 村口广场, 暴风雨刚过。你正在和一位路过帮忙的人【面对面说话】。\n本句自然带出: {beat}\n只输出直接说给对方听的台词, 禁止第一人称叙述, 禁止描写对方, 禁止内心描写, 禁止复述指令。"
        stub = ["(对方: 你还好吗？这雨太大了)", "(对方: 别难过，庆典也只是推迟到下周而已)",
                "(对方: 你手里那是…什么锄头？)", "(对方: 要不明天我来帮你收拾吧)"][i]
        # 最小上下文: 只给对方最近一句, 不给任何历史台词(人设+节拍已含全部信息)
        line = gen(system, f"对方刚说: {stub}\n请回一句。")
        history.append(line)
        lines.append(line)
    results[pname] = lines

for i, beat in enumerate(BEATS):
    print(f"\n── 节拍{i+1}: {beat[:24]}…")
    for pname, lines in results.items():
        print(f"  {pname}: {lines[i]}")
