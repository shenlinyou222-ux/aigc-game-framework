# -*- coding: utf-8 -*-
"""表情包 LoRA 数据集整理器: 抽卡原图 -> 可训练数据集。
流程: 扫描 raw/ -> sha256 去重 -> 主体裁方(有 alpha 用包围盒, 无则中心方裁)
      -> 规整到 --size 方图 -> 本地 Chinese-CLIP 自动情绪打标(复用 tag_stickers 词表)
      -> 输出 images/ + captions.jsonl + contact_sheet.png + manifest.json
用法:
  cd aigc_demo_framework
  E:\\py312\\python.exe sticker_lora\\prep_dataset.py                     # 默认 小灰/512
  E:\\py312\\python.exe sticker_lora\\prep_dataset.py --char 长夜月 --size 768
  E:\\py312\\python.exe sticker_lora\\prep_dataset.py --no-clip           # 只规整不打标(快)
就绪判据(写进 manifest): 去重后>=60 张; 单标签置信度>=0.25 占比>70%; 情绪覆盖>=12/22。
"""
import argparse
import hashlib
import io
import json
import os
import sys

from PIL import Image, ImageDraw, ImageFont

HERE = os.path.dirname(os.path.abspath(__file__))
FRAMEWORK = os.path.dirname(HERE)
sys.path.insert(0, FRAMEWORK)

Image.MAX_IMAGE_PIXELS = 200_000_000
EXTS = (".png", ".jpg", ".jpeg", ".webp")


def sha256_of(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def square_on_subject(img: Image.Image, size: int) -> Image.Image:
    """有 alpha: 按非透明包围盒裁方; 无 alpha: 短边中心方裁。再缩放到 size。"""
    img = img.convert("RGBA")
    w, h = img.size
    if img.mode == "RGBA" and img.getextrema()[3][0] < 255:
        bbox = img.getchannel("A").getbbox()
        if bbox:
            x0, y0, x1, y1 = bbox
            cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
            side = max(x1 - x0, y1 - y0) * 1.08  # 留 8% 呼吸边
            side = min(side, max(w, h))
    else:
        cx, cy = w / 2, h / 2
        side = min(w, h)
    half = side / 2
    left, top = max(0, round(cx - half)), max(0, round(cy - half))
    right, bottom = min(w, round(cx + half)), min(h, round(cy + half))
    crop = img.crop((left, top, right, bottom))
    side = max(crop.size)
    canvas = Image.new("RGBA", (side, side), (0, 0, 0, 0))
    canvas.paste(crop, ((side - crop.width) // 2, (side - crop.height) // 2))
    return canvas.resize((size, size), Image.LANCZOS)


def _feat(x):
    """transformers<5 的 get_*_features 直接返回张量; 5.x 返回 ModelOutput, 投影特征在 pooler_output."""
    return x.pooler_output if hasattr(x, "pooler_output") else x


def load_clip():
    import torch
    from transformers import ChineseCLIPModel, ChineseCLIPProcessor
    import tag_stickers as ts
    mp = os.path.join(FRAMEWORK, "models", "chinese-clip")
    model = ChineseCLIPModel.from_pretrained(mp).eval()
    proc = ChineseCLIPProcessor.from_pretrained(mp)
    with torch.no_grad():
        tv = proc(text=ts.PROMPTS, return_tensors="pt", padding=True)
        tvec = _feat(model.get_text_features(**tv))
        tvec = tvec / tvec.norm(dim=-1, keepdim=True)
    return model, proc, tvec, ts.EMOTIONS


def classify(model, proc, tvec, img: Image.Image, emotions, topk=3):
    import torch
    with torch.no_grad():
        iv = _feat(model.get_image_features(**proc(images=img.convert("RGB"), return_tensors="pt")))
    iv = iv / iv.norm(dim=-1, keepdim=True)
    sims = (iv @ tvec.T)[0]
    top = torch.topk(sims, topk)
    return [(emotions[j], round(float(s), 3)) for s, j in zip(top.values, top.indices)]


def contact_sheet(items, out_path: str, thumb=160, cols=8):
    rows = (len(items) + cols - 1) // cols
    sheet = Image.new("RGB", (cols * thumb, rows * (thumb + 18)), (24, 24, 24))
    draw = ImageDraw.Draw(sheet)
    try:
        font = ImageFont.truetype("C:/Windows/Fonts/msyh.ttc", 12)
    except OSError:
        font = ImageFont.load_default()
    for i, it in enumerate(items):
        x, y = (i % cols) * thumb, (i // cols) * (thumb + 18)
        im = Image.open(it["file"]).convert("RGBA")
        im.thumbnail((thumb, thumb))
        bg = Image.new("RGBA", (thumb, thumb), (40, 40, 40, 255))
        bg.paste(im, ((thumb - im.width) // 2, (thumb - im.height) // 2), im)
        sheet.paste(bg.convert("RGB"), (x, y))
        label = f'{it["index"]} {it["emotions"][0][0] if it["emotions"] else "?"}'
        draw.text((x + 2, y + thumb + 2), label, fill=(220, 220, 220), font=font)
    sheet.save(out_path)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=os.path.join(HERE, "raw"))
    ap.add_argument("--out", default=os.path.join(HERE, "dataset"))
    ap.add_argument("--char", default="小灰")
    ap.add_argument("--trigger", default="xiaohui_sticker")
    ap.add_argument("--size", type=int, default=512)
    ap.add_argument("--no-clip", action="store_true", help="跳过情绪打标(只规整)")
    args = ap.parse_args()

    files = sorted(
        os.path.join(r, f)
        for r, _, fs in os.walk(args.src)
        for f in fs
        if f.lower().endswith(EXTS)
    )
    if not files:
        print(f"[!] {args.src} 下没有图片。把抽卡原图丢进去再跑。")
        return

    seen, uniq = set(), []
    for f in files:
        d = sha256_of(f)
        if d not in seen:
            seen.add(d)
            uniq.append(f)
    dropped = len(files) - len(uniq)
    print(f"[+] 扫到 {len(files)} 张, 去重后 {len(uniq)} 张 (剔 {dropped} 重复)")

    img_dir = os.path.join(args.out, "images")
    os.makedirs(img_dir, exist_ok=True)
    clip = (None, None, None, None)
    if not args.no_clip:
        print("[+] 加载本地 Chinese-CLIP 情绪分类器 ...")
        clip = load_clip()

    items, captions = [], []
    for i, src in enumerate(uniq):
        try:
            img = Image.open(src)
            img.load()
        except Exception as e:
            print(f"  [skip] 无法读取 {src}: {e}")
            continue
        norm = square_on_subject(img, args.size)
        rel = f"{i:03d}.png"
        dst = os.path.join(img_dir, rel)
        norm.save(dst)
        emotions = classify(clip[0], clip[1], clip[2], norm, clip[3]) if not args.no_clip else []
        cap = f"{args.trigger}, {args.char}表情包, {emotions[0][0]}的表情包" if emotions else args.trigger
        items.append({"index": i, "file": dst, "emotions": emotions, "source": src})
        captions.append({
            "file": rel, "caption": cap, "emotions": emotions,
            "source": os.path.relpath(src, FRAMEWORK).replace("\\", "/"),
        })

    with open(os.path.join(args.out, "captions.jsonl"), "w", encoding="utf-8") as f:
        for c in captions:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")
    contact_sheet(items, os.path.join(args.out, "contact_sheet.png"))

    coverage = {}
    for it in items:
        for name, score in it["emotions"]:
            if score >= 0.25:
                coverage[name] = coverage.get(name, 0) + 1
                break
    confident = sum(1 for it in items
                    if it["emotions"] and it["emotions"][0][1] >= 0.25)
    manifest = {
        "char": args.char, "trigger": args.trigger, "size": args.size,
        "total_raw": len(files), "duplicates": dropped, "usable": len(items),
        "confident_ratio": round(confident / max(1, len(items)), 3),
        "emotion_coverage": coverage,
        "ready": bool(len(items) >= 60 and confident / max(1, len(items)) > 0.7
                      and len(coverage) >= 12),
        "hint": "缺口情绪优先定向补抽; ready=true 即可进入 LoRA 微调",
    }
    with open(os.path.join(args.out, "manifest.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    print(f"[+] images/{len(items)} 张 @ {args.size}px, captions.jsonl, contact_sheet.png")
    print(f"[+] 情绪覆盖 {len(coverage)}/22 类, 高置信占比 {manifest['confident_ratio']}")
    print(f"[{'✅' if manifest['ready'] else '⏳'}] 就绪: {manifest['ready']}  (标准>=60张/置信>70%/覆盖>=12类)")


if __name__ == "__main__":
    main()
