#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

"""
metrics.py — 评估 3DGS 渲染结果

M0.1 (2026-07-25) 升级:
  - 除了原有 PSNR / SSIM / LPIPS (全图), 新增 3 项前景/背景分离指标:
      PSNR_fg / SSIM_fg / LPIPS_fg / PSNR_bg
  - 分离用的 mask 从 <source_path>/weight_maps/<name>.(png|npy) 加载
  - 若找不到 wm, 自动回退到只输出全图 3 指标, 兼容旧行为
  - mask 二值化阈值默认 0.5, 可通过 --wm_thr 调整

用法:
    python metrics.py -m <model_path>              # 只出 3 指标 (若无 wm)
    python metrics.py -m <model_path> --wm_dir /path/to/weight_maps
    python metrics.py -m <model_path> --wm_thr 0.3
"""

from pathlib import Path
import os
from PIL import Image
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as tf
from utils.loss_utils import ssim
from utils.metrics_fg import masked_psnr, bg_psnr, masked_ssim, masked_lpips, mask_coverage
from lpipsPyTorch import lpips
import json
from tqdm import tqdm
from utils.image_utils import psnr
from argparse import ArgumentParser


def _load_wm(wm_dir: Path, name: str, target_hw: tuple, alias_stem: str = None):
    """
    尝试从 wm_dir 加载与 `name` 同名的权重图.
    返回 (1, H, W) float tensor in cuda, 或 None.
    支持 .npy / .png / .jpg. 会 resize 到 target_hw.
    若同名找不到, 用 alias_stem (由外部按 test 顺序映射到原图 stem, 例如
    "00000" -> "0001") 再试一次.
    """
    if wm_dir is None or not wm_dir.is_dir():
        return None
    stems_try = [os.path.splitext(name)[0]]
    if alias_stem is not None and alias_stem not in stems_try:
        stems_try.append(alias_stem)
    for stem in stems_try:
        # npy 精度更高, 优先
        p = wm_dir / (stem + ".npy")
        if p.exists():
            arr = np.load(str(p)).astype(np.float32)
            return _resize_wm_to_tensor(arr, target_hw)
        for ext in [".png", ".jpg", ".jpeg"]:
            p = wm_dir / (stem + ext)
            if p.exists():
                im = np.array(Image.open(p).convert("L"), dtype=np.float32) / 255.0
                return _resize_wm_to_tensor(im, target_hw)
    return None


def _build_test_alias_map(scene_dir: str, n_test: int) -> list:
    """
    3DGS render 时 test 图按 enumerate(test_views) 顺序命名为 "00000.png" 等,
    但预计算的 wm 用的是原图 stem (例如 "0001"). 这里重建映射:
        输出 aliases[i] = 原图 stem, 对应 test 第 i 张.
    规则: 读 cfg_args 拿 source_path -> sparse/0/images.bin -> sorted by name
          -> llffhold=8 取 [0, 8, 16, ...] 张 -> stem 去后缀.
    找不到时返回 [] (metrics.py 会走同名路径).
    """
    try:
        import re
        cfg = Path(scene_dir) / "cfg_args"
        if not cfg.exists():
            return []
        m = re.search(r"source_path=['\"]([^'\"]+)['\"]", cfg.read_text())
        if not m:
            return []
        src = m.group(1)
        from scene.colmap_loader import read_extrinsics_binary
        extrs = read_extrinsics_binary(os.path.join(src, "sparse", "0", "images.bin"))
        names_sorted = sorted([e.name for e in extrs.values()])
        # 与 dataset_readers.py 里 llffhold=8 保持一致
        test_names = [n for i, n in enumerate(names_sorted) if i % 8 == 0]
        if len(test_names) != n_test:
            print(f"  [warn] alias map: sparse test={len(test_names)}, render test={n_test}, 用截取")
        aliases = [os.path.splitext(n)[0] for n in test_names[:n_test]]
        return aliases
    except Exception as e:
        print(f"  [warn] 无法建立 test alias map: {e}")
        return []


def _resize_wm_to_tensor(arr: np.ndarray, target_hw: tuple):
    H, W = target_hw
    t = torch.from_numpy(arr).float()
    if t.ndim == 3:
        t = t[..., 0]
    t = t.unsqueeze(0).unsqueeze(0)  # (1, 1, h, w)
    if t.shape[-2] != H or t.shape[-1] != W:
        t = F.interpolate(t, size=(H, W), mode="bilinear", align_corners=False)
    return t.squeeze(0).cuda()  # (1, H, W)


def readImages(renders_dir, gt_dir):
    renders = []
    gts = []
    image_names = []
    for fname in sorted(os.listdir(renders_dir)):
        render = Image.open(renders_dir / fname)
        gt = Image.open(gt_dir / fname)
        renders.append(tf.to_tensor(render).unsqueeze(0)[:, :3, :, :].cuda())
        gts.append(tf.to_tensor(gt).unsqueeze(0)[:, :3, :, :].cuda())
        image_names.append(fname)
    return renders, gts, image_names


def evaluate(model_paths, wm_dir: str = None, wm_thr: float = 0.5):

    full_dict = {}
    per_view_dict = {}
    full_dict_polytopeonly = {}
    per_view_dict_polytopeonly = {}
    print("")

    for scene_dir in model_paths:
        try:
            print("Scene:", scene_dir)
            full_dict[scene_dir] = {}
            per_view_dict[scene_dir] = {}
            full_dict_polytopeonly[scene_dir] = {}
            per_view_dict_polytopeonly[scene_dir] = {}

            test_dir = Path(scene_dir) / "test"

            # ── wm dir 解析: 优先 --wm_dir, 其次 cfg_args 里的 source_path/weight_maps ──
            wm_dir_path = None
            if wm_dir:
                wm_dir_path = Path(wm_dir)
            else:
                cfg_path = Path(scene_dir) / "cfg_args"
                if cfg_path.exists():
                    try:
                        cfg_str = cfg_path.read_text()
                        # simple parse: 找 source_path='...'
                        import re
                        m = re.search(r"source_path=['\"]([^'\"]+)['\"]", cfg_str)
                        if m:
                            candidate = Path(m.group(1)) / "weight_maps"
                            if candidate.is_dir():
                                wm_dir_path = candidate
                    except Exception as e:
                        print(f"  [warn] parse cfg_args failed: {e}")
            if wm_dir_path is not None and wm_dir_path.is_dir():
                print(f"  wm_dir = {wm_dir_path} (thr={wm_thr})")
            else:
                print(f"  wm_dir 未找到, 仅输出全图 PSNR/SSIM/LPIPS")

            for method in os.listdir(test_dir):
                print("Method:", method)

                full_dict[scene_dir][method] = {}
                per_view_dict[scene_dir][method] = {}
                full_dict_polytopeonly[scene_dir][method] = {}
                per_view_dict_polytopeonly[scene_dir][method] = {}

                method_dir = test_dir / method
                gt_dir = method_dir/ "gt"
                renders_dir = method_dir / "renders"
                renders, gts, image_names = readImages(renders_dir, gt_dir)

                # 构造 test 图 stem 别名映射: 00000.png -> 0001 etc.
                alias_map = _build_test_alias_map(scene_dir, len(image_names))
                if alias_map:
                    print(f"  test alias: {image_names[0]} -> {alias_map[0]}, ... "
                          f"{image_names[-1]} -> {alias_map[-1]} ({len(alias_map)} 张)")

                ssims = []
                psnrs = []
                lpipss = []
                psnrs_fg, psnrs_bg = [], []
                ssims_fg, lpipss_fg = [], []
                covs = []

                for idx in tqdm(range(len(renders)), desc="Metric evaluation progress"):
                    r, g = renders[idx], gts[idx]
                    ssims.append(ssim(r, g).item())
                    psnrs.append(psnr(r, g).item())
                    lpipss.append(lpips(r, g, net_type='vgg').item())

                    # ── fg/bg 分离 (需要 wm) ──
                    if wm_dir_path is not None:
                        H, W = r.shape[-2], r.shape[-1]
                        alias = alias_map[idx] if idx < len(alias_map) else None
                        wm = _load_wm(wm_dir_path, image_names[idx], (H, W),
                                      alias_stem=alias)
                        if wm is not None:
                            cov = mask_coverage(wm, thr=wm_thr)
                            covs.append(cov)
                            psnr_fg_v = masked_psnr(r, g, wm, thr=wm_thr).item()
                            psnr_bg_v = bg_psnr(r, g, wm, thr=wm_thr).item()
                            psnrs_fg.append(psnr_fg_v)
                            psnrs_bg.append(psnr_bg_v)
                            # SSIM/LPIPS masked (贵一点, 但仍可接受)
                            ssim_fg_v = masked_ssim(r, g, wm, thr=wm_thr).item()
                            lpips_fg_v = masked_lpips(r, g, wm, thr=wm_thr, net_type='vgg').item()
                            ssims_fg.append(ssim_fg_v)
                            lpipss_fg.append(lpips_fg_v)

                def _mean_ignore_nan(xs):
                    xs = [x for x in xs if x == x]  # drop nan
                    if not xs:
                        return None
                    return float(np.mean(xs))

                m_ssim = _mean_ignore_nan(ssims)
                m_psnr = _mean_ignore_nan(psnrs)
                m_lpips = _mean_ignore_nan(lpipss)
                m_psnr_fg = _mean_ignore_nan(psnrs_fg)
                m_psnr_bg = _mean_ignore_nan(psnrs_bg)
                m_ssim_fg = _mean_ignore_nan(ssims_fg)
                m_lpips_fg = _mean_ignore_nan(lpipss_fg)
                m_cov = _mean_ignore_nan(covs)

                # ── 打印 5 列表 ──
                def _f(x, w=12):
                    if x is None:
                        return "-".rjust(w)
                    return f"{x:>{w}.7f}"

                print(f"  PSNR_all : {_f(m_psnr)}  SSIM : {_f(m_ssim)}  LPIPS : {_f(m_lpips)}")
                if m_psnr_fg is not None:
                    print(f"  PSNR_fg  : {_f(m_psnr_fg)}  SSIM_fg: {_f(m_ssim_fg)}  LPIPS_fg: {_f(m_lpips_fg)}")
                    print(f"  PSNR_bg  : {_f(m_psnr_bg)}  fg_cov : {_f(m_cov)}")
                print("")

                agg = {"SSIM": m_ssim, "PSNR": m_psnr, "LPIPS": m_lpips}
                if m_psnr_fg is not None:
                    agg.update({
                        "PSNR_fg": m_psnr_fg, "PSNR_bg": m_psnr_bg,
                        "SSIM_fg": m_ssim_fg, "LPIPS_fg": m_lpips_fg,
                        "fg_coverage": m_cov, "wm_thr": wm_thr,
                    })
                full_dict[scene_dir][method].update(agg)

                per_view = {"SSIM": dict(zip(image_names, ssims)),
                            "PSNR": dict(zip(image_names, psnrs)),
                            "LPIPS": dict(zip(image_names, lpipss))}
                if psnrs_fg:
                    per_view.update({
                        "PSNR_fg": dict(zip(image_names, psnrs_fg)),
                        "PSNR_bg": dict(zip(image_names, psnrs_bg)),
                        "SSIM_fg": dict(zip(image_names, ssims_fg)),
                        "LPIPS_fg": dict(zip(image_names, lpipss_fg)),
                        "fg_coverage": dict(zip(image_names, covs)),
                    })
                per_view_dict[scene_dir][method].update(per_view)

            with open(scene_dir + "/results.json", 'w') as fp:
                json.dump(full_dict[scene_dir], fp, indent=True)
            with open(scene_dir + "/per_view.json", 'w') as fp:
                json.dump(per_view_dict[scene_dir], fp, indent=True)
        except Exception as e:
            import traceback
            print("Unable to compute metrics for model", scene_dir)
            traceback.print_exc()

if __name__ == "__main__":
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)

    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    parser.add_argument('--model_paths', '-m', required=True, nargs="+", type=str, default=[])
    parser.add_argument('--wm_dir', default=None, type=str,
                        help="weight_maps 目录, 提供后启用 fg/bg 分离指标. 不给则自动从 cfg_args 的 source_path/weight_maps 读.")
    parser.add_argument('--wm_thr', default=0.5, type=float,
                        help="mask 二值化阈值 (默认 0.5)")
    args = parser.parse_args()
    evaluate(args.model_paths, wm_dir=args.wm_dir, wm_thr=args.wm_thr)
