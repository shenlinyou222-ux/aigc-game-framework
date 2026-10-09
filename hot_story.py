# -*- coding: utf-8 -*-
"""剧情对话管线: qwen3.5 产出完整情节 YAML -> 确定性转换 -> story_dialog 命令
猫自动跳台词讲完一段故事, 结束后送主角一个道具(主角全程零输入)。
用法: python hot_story.py [npc] [剧情主题]
"""
import json, os, sys, time, urllib.request

from hot_dialog_pipeline import OLLAMA, MODEL, INBOX, sanitize

STORY_CHARS = 24          # 剧情单句可长一点
GIFT = {"item_id": "cat_bell", "name": "猫铃铛",
        "icon_path": "res://graphics/icons/apple.png", "count": 1}


def gen_story_yaml(npc: str, theme: str) -> str:
    sys_p = (
        "你是游戏剧情编剧。写一段 NPC 对主角讲述的完整小剧情(6-8句台词, 独白推进, 主角不说话), "
        "有起因-经过-结尾, 结尾要表达谢意并提到要送一件小礼物。"
        "严格按示例格式只输出 YAML, 不要代码块不要解释:\n"
        "npc: 猫\n"
        "lines:\n"
        "  - <第1句>\n"
        "  - <第2句>\n"
        f"每句不超过{STORY_CHARS}个汉字。"
    )
    body = {"model": MODEL, "messages": [
        {"role": "system", "content": sys_p},
        {"role": "user", "content": f"角色: {npc}(傲娇又重情的猫)。剧情主题: {theme}"}],
        "stream": False, "think": False, "options": {"temperature": 0.85}}
    req = urllib.request.Request(OLLAMA,
        data=json.dumps(body, ensure_ascii=False).encode("utf-8"),
        headers={"Content-Type": "application/json"})
    return json.load(urllib.request.urlopen(req, timeout=180))["message"]["content"]


def story_to_command(yaml_text: str, npc: str, actor: str, gift: dict) -> dict:
    from hot_dialog_pipeline import _parse_yaml_light
    n, lines = _parse_yaml_light(yaml_text)
    lines = [re_norm(l) for l in lines]
    lines = [l for l in lines if l][:8]
    if len(lines) < 3:
        raise ValueError(f"剧情台词不足3句: {lines}")
    return {"command_id": f"story_{int(time.time()*1000)}",
            "command_type": "story_dialog", "actor": actor,
            "params": {"npc": npc, "lines": lines, "gift": gift}}   # npc 以调用方为准, 不采信模型


def re_norm(l: str) -> str:
    import re
    l = re.sub(r"\s+", " ", l).strip().strip("'\"")
    # 过滤模板占位符复读: <第1句> / <第N句> / <...>
    if re.search(r"<[^>]*>", l) or re.fullmatch(r"第?\s*[0-9一二三四五六七八九十]+\s*句.*", l):
        return ""
    return l[:STORY_CHARS]


if __name__ == "__main__":
    npc = sys.argv[1] if len(sys.argv) > 1 else "Cat"
    theme = sys.argv[2] if len(sys.argv) > 2 else "猫说起暴风雨那晚自己被困在鸡舍, 是主角修好了屋顶救了它"
    y = ""
    for attempt in range(2):
        y = gen_story_yaml(npc, theme)
        print("== 阶段1 qwen 剧情YAML (attempt %d) ==\n%s" % (attempt + 1, y.strip()))
        try:
            cmd = story_to_command(y, npc, MODEL, GIFT)
            break
        except ValueError:
            if attempt == 1:
                raise
    print("\n== 阶段2 确定性转换(剧情+gift) ==\n" + json.dumps(cmd, ensure_ascii=False, indent=2))
    os.makedirs(os.path.dirname(INBOX), exist_ok=True)
    with open(INBOX, "a", encoding="utf-8") as f:
        f.write(json.dumps(cmd, ensure_ascii=False) + "\n")
    print("\n== 阶段3 已提交 story_dialog ->", INBOX)
