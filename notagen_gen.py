# -*- coding: utf-8 -*-
"""NotaGen-X 运行器: fp16 装载(~3.1GB显存) + prompt条件生成 ABC 乐谱
用法:
  python notagen_gen.py "Romantic" "Schumann" "Piano" out.abc
"""
import os, sys, time

# NotaGen-X 上游仓库的本地检出目录与权重文件。上游代码与权重不入库：
# 用环境变量 NOTAGEN_DIR / NOTAGEN_WEIGHTS 指定，
# 默认放在本仓库的 third_party/NotaGen/ 下。
_HERE = os.path.dirname(os.path.abspath(__file__))
NOTAGEN = os.environ.get("NOTAGEN_DIR", os.path.join(_HERE, "third_party", "NotaGen", "gradio"))
WEIGHTS = os.environ.get("NOTAGEN_WEIGHTS", os.path.join(_HERE, "third_party", "NotaGen", "weights_notagenx.pth"))


def main():
    os.chdir(NOTAGEN)
    sys.path.insert(0, NOTAGEN)

    import torch
    _orig_load = torch.load

    def _load_fp16(f, *a, **k):
        k.setdefault("map_location", "cpu")
        ckpt = _orig_load(f, *a, **k)
        sd = ckpt.get("model", ckpt) if isinstance(ckpt, dict) else ckpt
        for key in list(sd):
            if torch.is_tensor(sd[key]) and sd[key].is_floating_point():
                sd[key] = sd[key].half()
        return {"model": sd} if isinstance(ckpt, dict) and "model" in ckpt else sd

    torch.load = _load_fp16   # 拦截 fp32 checkpoint -> fp16

    import inference as ng    # 模块级完成建模+装载(fp16)
    ng.model = ng.model.to(torch.device("cuda")).eval()
    free, total = torch.cuda.mem_get_info()
    print(f"[notagen] model loaded fp16, VRAM: {(total-free)/1e9:.2f}GB used / {total/1e9:.2f}GB", flush=True)

    period, composer, instr = sys.argv[1], sys.argv[2], sys.argv[3]
    out_path = sys.argv[4] if len(sys.argv) > 4 else "gen.abc"

    t0 = time.time()
    abc_code = ng.inference_patch(period, composer, instr)
    dt = time.time() - t0
    if isinstance(abc_code, str) and len(abc_code) > 50:
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(abc_code)
        print(f"\n[notagen] OK {dt:.1f}s -> {out_path} ({len(abc_code)} chars)")
    else:
        print(f"\n[notagen] FAILED in {dt:.1f}s, ret={str(abc_code)[:100]}")
        sys.exit(1)


if __name__ == "__main__":
    main()
