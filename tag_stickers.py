# -*- coding: utf-8 -*-
"""表情包情绪打标器(Chinese-CLIP 零样本, 只打指定目录)
用法:
  python tag_stickers.py "E:\stickers\ZZZ\邦布"          # 打标一个目录
产物:
  sticker_tags.json  {路径: {"emotions": [(标签,分),...], "character": 目录名}}
  sticker_review.html 左右对照浏览页(浏览器打开人工复核)
"""
import glob, json, os, sys

EMOTIONS = [
    "开心大笑", "微笑开心", "生气发怒", "委屈巴巴", "大哭流泪", "惊讶震惊",
    "害羞脸红", "疑惑不解", "无语呆滞", "坏笑偷笑", "害怕发抖", "得意得意",
    "可爱卖萌", "比心感谢", "点赞加油", "石化懵了", "期待渴望", "困倦睡觉",
    "吃东西", "敬礼认真", "委屈生气", "冷笑蔑视",
]
PROMPTS = [e + "的表情包" for e in EMOTIONS]
TAGS_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "sticker_tags.json")


def tag_dir(directory: str) -> list[dict]:
    import torch
    from PIL import Image
    from transformers import ChineseCLIPModel, ChineseCLIPProcessor
    mp = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models", "chinese-clip")
    m = ChineseCLIPModel.from_pretrained(mp).eval()
    p = ChineseCLIPProcessor.from_pretrained(mp)
    files = sorted(f for f in glob.glob(os.path.join(directory, "**", "*"), recursive=True)
                   if f.lower().endswith((".png", ".jpg", ".webp")))
    print(f"待打标: {len(files)} 张 @ {directory}")
    def _feat(x):  # transformers 5.x 兼容: get_*_features 返回 ModelOutput, 特征在 pooler_output
        return x.pooler_output if hasattr(x, "pooler_output") else x
    with torch.no_grad():
        tf = p(text=PROMPTS, return_tensors="pt", padding=True)
        tvec = _feat(m.get_text_features(**tf)); tvec = tvec / tvec.norm(dim=-1, keepdim=True)
        results = []
        for i, f in enumerate(files):
            try:
                img = Image.open(f).convert("RGB")
            except Exception:
                continue
            iv = _feat(m.get_image_features(**p(images=img, return_tensors="pt")))
            iv = iv / iv.norm(dim=-1, keepdim=True)
            sims = (iv @ tvec.T)[0]
            top = torch.topk(sims, 3)
            results.append({
                "file": f.replace("\\", "/"),
                "character": os.path.basename(os.path.dirname(f)),
                "emotions": [[EMOTIONS[j], round(float(s), 3)]
                             for s, j in zip(top.values, top.indices)],
            })
            if (i + 1) % 20 == 0:
                print(f"  {i+1}/{len(files)}")
    tags = {}
    if os.path.exists(TAGS_FILE):
        tags = json.load(open(TAGS_FILE, encoding="utf-8"))
    for r in results:
        tags[r["file"]] = {"emotions": r["emotions"], "character": r["character"]}
    json.dump(tags, open(TAGS_FILE, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(f"完成, 累计标注 {len(tags)} 张 -> {os.path.basename(TAGS_FILE)}")
    return results


def write_html(results: list[dict]) -> None:
    rows = "".join(
        f'<div class="card"><img src="file:///{r["file"]}">'
        f'<div class="tags">{ " | ".join(f"{e}({s})" for e, s in r["emotions"]) }</div></div>'
        for r in results)
    html = ("""<html><meta charset="utf-8"><style>
    body{background:#222;color:#ddd;font-family:sans-serif}
    .grid{display:flex;flex-wrap:wrap;gap:12px}
    .card{width:220px;background:#333;border-radius:8px;padding:8px;text-align:center}
    img{max-width:200px;max-height:200px;border-radius:6px}
    .tags{font-size:13px;margin-top:6px;color:#ffd21f}</style><body>
    <h2>表情打标复核 (%d 张, 第一标签为预测情绪)</h2><div class="grid">%s</div></body></html>"""
        % (len(results), rows))
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "sticker_review.html")
    open(out, "w", encoding="utf-8").write(html)
    print("复核页 ->", out)


if __name__ == "__main__":
    d = sys.argv[1] if len(sys.argv) > 1 else r"E:\stickers\ZZZ\邦布"
    res = tag_dir(d)
    write_html(res)
