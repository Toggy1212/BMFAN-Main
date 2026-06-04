import argparse
import importlib
import os
import sys
import torch
from fvcore.nn import FlopCountAnalysis


def parse_scales(scale_str: str):
    """
    支持:
      --scales 2,3,4
      --scales 2
    """
    scale_str = scale_str.replace("+", ",")
    items = [s.strip() for s in scale_str.split(",") if s.strip()]
    return [int(s) for s in items]


def count_params_k(model: torch.nn.Module) -> float:
    """Params in K."""
    return sum(p.numel() for p in model.parameters()) / 1e3


def count_flops_g_fvcore_total(model: torch.nn.Module, lr_h: int, lr_w: int, device: torch.device) -> float:
    """
    严格按你参考代码的口径：
      FLOPs(G) = FlopCountAnalysis(model, x).total() / 1e9
    """
    model = model.to(device)
    model.eval()
    x = torch.randn(1, 3, lr_h, lr_w, device=device)
    with torch.no_grad():
        total = FlopCountAnalysis(model, x).total()
    return total / 1e9


def build_model_by_name(args):
    """
    从 EDSR-PyTorch 的 src/model/<name>.py 动态加载并调用 make_model(args)
    - 这样你可以 --model ERIRAN / ERIRAN_block7 / BMFAN 等
    - 前提：对应的 model/<name>.py 内存在 make_model(args)
    """
    model_name = args.model.lower()
    try:
        module = importlib.import_module(f"model.{model_name}")
    except Exception as e:
        raise RuntimeError(
            f"❌ 无法导入 model.{model_name}。\n"
            f"请确认：\n"
            f"  1) 你在 EDSR-PyTorch-master/src 目录下运行\n"
            f"  2) src/model/{model_name}.py 存在\n"
            f"原始错误：{e}"
        )

    if not hasattr(module, "make_model"):
        raise RuntimeError(
            f"❌ model.{model_name} 中找不到 make_model(args)。\n"
            f"请把该模型文件改成 EDSR 框架标准接口：提供 make_model(args)。"
        )

    return module.make_model(args)


def main():
    parser = argparse.ArgumentParser()

    # 指定模型：对应 src/model/<model>.py
    parser.add_argument("--model", type=str, required=True,
                        help="模型名（对应 src/model/<model>.py），如 ERIRAN / ERIRAN_block7 / BMFAN")

    # 目标 HR 与 scales（按 HR 推 LR）
    parser.add_argument("--hr_h", type=int, default=720, help="目标 HR 高度（默认 720）")
    parser.add_argument("--hr_w", type=int, default=1280, help="目标 HR 宽度（默认 1280）")
    parser.add_argument("--scales", type=str, default="2,3,4", help="要统计的倍率，如 2,3,4 或 2")

    # 常见模型参数（尽量兼容 EDSR 框架）
    parser.add_argument("--n_feats", type=int, default=48)
    parser.add_argument("--n_colors", type=int, default=3)
    parser.add_argument("--rgb_range", type=int, default=255)
    parser.add_argument("--n_resblocks", type=int, default=16)
    parser.add_argument("--res_scale", type=float, default=1.0)

    # device
    parser.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"])

    args = parser.parse_args()

    # 确保能 import model.*
    cur_dir = os.path.dirname(os.path.abspath(__file__))
    if cur_dir not in sys.path:
        sys.path.insert(0, cur_dir)

    device = torch.device("cuda" if (args.device == "cuda" and torch.cuda.is_available()) else "cpu")

    scales = parse_scales(args.scales)

    print("=" * 112)
    print(f"Model Params & FLOPs Test (fvcore.total()/1e9) | Target HR ≈ {args.hr_h}x{args.hr_w}")
    print(f"Model: {args.model} | Device: {device.type}")
    print("LR is derived by integer division: LR_H = HR_H // scale, LR_W = HR_W // scale")
    print("=" * 112)
    print(f"{'Scale':<6} | {'LR (H x W)':<22} | {'Params (K)':<14} | {'FLOPs (G)':<14}")
    print("-" * 112)

    for s in scales:
        lr_h = args.hr_h // s
        lr_w = args.hr_w // s

        approx_flag = ""
        if (args.hr_h % s != 0) or (args.hr_w % s != 0):
            approx_flag = " (approx)"

        # EDSR 框架里 args.scale 通常是 list[int]
        # 这里每个 scale 单独构建一次模型，确保 scale-dependent 的上采样层正确
        args.scale = [s]

        model = build_model_by_name(args)
        model.eval()

        params_k = count_params_k(model)
        flops_g = count_flops_g_fvcore_total(model, lr_h, lr_w, device)

        lr_str = f"{lr_h}x{lr_w}{approx_flag}"
        print(f"x{s:<5} | {lr_str:<22} | {params_k:>12.2f} K | {flops_g:>12.3f} G")

    print("=" * 112)
    print("Note:")
    print(" - FLOPs(G) strictly follows your reference: FlopCountAnalysis(model, x).total() / 1e9")
    print(" - Params(K) does not depend on resolution; FLOPs(G) scales with LR HxW.")
    print(" - For x3 (720x1280 not divisible by 3), LR uses integer division and is marked (approx).")
    print("=" * 112)


if __name__ == "__main__":
    main()
