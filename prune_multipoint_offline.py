#!/usr/bin/env python
"""
Offline hard-prune (方案 A · 多点椭圆采样) 离线版
============================================
不介入训练, 只对已训练好的 GS 场景做一次性筛除, 用来纯粹检验
"多点椭圆采样能否识别真正该剔的雾状/骑墙/横穿型高斯"

用法:
  python prune_multipoint_offline.py \
      -m /path/to/model_dir \
      --iteration 30000 \
      --wm_thr 0.3 \
      --grid 5 --radius_mul 2.0 --overflow_ratio 0.7 \
      [--output_iter 30001]      # 保存到 point_cloud/iteration_30001/
      [--dry_run]                # 只统计, 不保存

之后:
  1. 拿 render.py 分别 render 原 iteration 和 output_iter, 对比 test 集 PSNR
  2. 或直接用本脚本 --with_metrics 自动跑
"""
import os, sys, argparse
import torch
import numpy as np

# 让脚本能找到 gaussian-splatting 的模块
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from arguments import ModelParams, PipelineParams
from argparse import ArgumentParser
from scene import Scene, GaussianModel
from gaussian_renderer import render
from utils.weight_map_utils import WeightMapLoader

try:
    from diff_gaussian_rasterization import SparseGaussianAdam
    SPARSE_ADAM_AVAILABLE = True
except:
    SPARSE_ADAM_AVAILABLE = False


def offline_multipoint_prune(gaussians, scene, weight_map_loader, pipe, background,
                              thr, grid_size, radius_mul, overflow_ratio,
                              train_test_exp=False):
    """
    等价于 train.py 里 wm_hard_prune_step, 但作为一次性调用.
    返回: prune_mask (bool tensor, True=该剔), stats dict
    """
    with torch.no_grad():
        N = gaussians.get_xyz.shape[0]
        device = gaussians.get_xyz.device
        min_overflow = torch.ones(N, device=device)
        visible_in_any = torch.zeros(N, dtype=torch.bool, device=device)

        train_cams = scene.getTrainCameras()
        xyz = gaussians.get_xyz
        ones_col = torch.ones(N, 1, device=device)
        xyz_h = torch.cat([xyz, ones_col], dim=1)

        offs = torch.linspace(-radius_mul, radius_mul, grid_size, device=device)
        oy, ox = torch.meshgrid(offs, offs, indexing="ij")
        ox_flat, oy_flat = ox.reshape(-1), oy.reshape(-1)

        print(f"[offline-prune] scanning {len(train_cams)} views, "
              f"grid={grid_size}x{grid_size}, radius_mul={radius_mul}, "
              f"overflow_ratio={overflow_ratio}, wm_thr={thr}")

        for i, cam in enumerate(train_cams):
            wm = weight_map_loader.get(cam, device=device)   # (1, H, W)
            H, W = wm.shape[-2], wm.shape[-1]

            render_pkg = render(cam, gaussians, pipe, background,
                                 use_trained_exp=train_test_exp,
                                 separate_sh=SPARSE_ADAM_AVAILABLE)
            radii = render_pkg["radii"].float()

            proj = xyz_h @ cam.full_proj_transform
            w_h = proj[:, 3]
            in_front = w_h > 1e-6
            ndc = proj[:, :2] / (w_h.unsqueeze(-1) + 1e-8)
            in_ndc = (ndc[:, 0].abs() < 1) & (ndc[:, 1].abs() < 1)
            on_screen = in_front & in_ndc & (radii > 0)
            if not on_screen.any():
                continue

            idx = on_screen.nonzero(as_tuple=True)[0]
            cx = (ndc[idx, 0] + 1.0) * 0.5 * W
            cy = (ndc[idx, 1] + 1.0) * 0.5 * H
            r = radii[idx].clamp(min=1.0)

            sx = (cx.unsqueeze(1) + r.unsqueeze(1) * ox_flat.unsqueeze(0)).long().clamp(0, W - 1)
            sy = (cy.unsqueeze(1) + r.unsqueeze(1) * oy_flat.unsqueeze(0)).long().clamp(0, H - 1)
            w_samples = wm[0, sy, sx]

            overflow = (w_samples < thr).float().mean(dim=1)
            min_overflow[idx] = torch.minimum(min_overflow[idx], overflow)
            visible_in_any[idx] = True

            if (i + 1) % 50 == 0:
                print(f"  [{i+1}/{len(train_cams)}] "
                      f"cumulative visible={int(visible_in_any.sum())}/{N}")

        prune_mask = visible_in_any & (min_overflow > overflow_ratio)
        stats = {
            "N_total": N,
            "N_visible": int(visible_in_any.sum()),
            "N_never_visible": int((~visible_in_any).sum()),
            "N_prune": int(prune_mask.sum()),
            "N_keep": int(N - prune_mask.sum()),
            "prune_ratio": float(prune_mask.sum() / N),
        }
        # 外溢占比分布 (仅在可见的高斯里看)
        vis_ov = min_overflow[visible_in_any].cpu().numpy()
        if len(vis_ov) > 0:
            stats["overflow_p50"] = float(np.percentile(vis_ov, 50))
            stats["overflow_p90"] = float(np.percentile(vis_ov, 90))
            stats["overflow_p99"] = float(np.percentile(vis_ov, 99))
        return prune_mask, stats


def main():
    ap = ArgumentParser(description="Offline multipoint hard-prune (方案 A)")
    model_g = ModelParams(ap, sentinel=True)
    pipe_g = PipelineParams(ap)
    ap.add_argument("--iteration", type=int, default=30000,
                    help="加载哪个 iteration 的 ply")
    ap.add_argument("--output_iter", type=int, default=-1,
                    help="剪枝后保存到 point_cloud/iteration_<output_iter>/, "
                         "默认 iteration+1")
    ap.add_argument("--wm_thr", type=float, default=0.3,
                    help="权重图阈值 (wm<thr 视为非重点区)")
    ap.add_argument("--grid", type=int, default=5)
    ap.add_argument("--radius_mul", type=float, default=2.0)
    ap.add_argument("--overflow_ratio", type=float, default=0.7)
    ap.add_argument("--wm_mode", type=str, default="online")
    ap.add_argument("--wm_alpha", type=float, default=1.0)
    ap.add_argument("--wm_beta", type=float, default=1.0)
    ap.add_argument("--dry_run", action="store_true", help="只统计不保存")
    args = ap.parse_args(sys.argv[1:])
    dataset = model_g.extract(args)
    pipe = pipe_g.extract(args)

    # sentinel=True 下未传的字段会是 None, 需要用默认值补齐, 避免下游 None 报错
    _defaults = {
        "sh_degree": 3, "images": "images", "depths": "",
        "resolution": -1, "white_background": False,
        "train_test_exp": False, "data_device": "cuda", "eval": True,
    }
    for k, v in _defaults.items():
        if getattr(dataset, k, None) is None:
            setattr(dataset, k, v)

    if args.output_iter < 0:
        args.output_iter = args.iteration + 1

    print(f"[offline-prune] model_path = {dataset.model_path}")
    print(f"[offline-prune] source_path = {dataset.source_path}")
    print(f"[offline-prune] loading iteration {args.iteration}")

    # 加载场景
    gaussians = GaussianModel(dataset.sh_degree)
    scene = Scene(dataset, gaussians, load_iteration=args.iteration, shuffle=False)
    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    # 加载权重图
    weight_map_loader = WeightMapLoader(
        source_path=dataset.source_path,
        mode=args.wm_mode,
        alpha=args.wm_alpha,
        beta=args.wm_beta,
    )
    print(f"[offline-prune] weightmap: mode={args.wm_mode}, "
          f"alpha={args.wm_alpha}, beta={args.wm_beta}")

    # 跑筛选
    prune_mask, stats = offline_multipoint_prune(
        gaussians, scene, weight_map_loader, pipe, background,
        thr=args.wm_thr, grid_size=args.grid,
        radius_mul=args.radius_mul, overflow_ratio=args.overflow_ratio,
        train_test_exp=dataset.train_test_exp,
    )

    print("\n" + "=" * 60)
    print("=== Offline Prune Summary ===")
    print(f"  N_total          = {stats['N_total']}")
    print(f"  N_visible        = {stats['N_visible']}")
    print(f"  N_never_visible  = {stats['N_never_visible']}  (保护, 不剔)")
    print(f"  N_prune          = {stats['N_prune']}  "
          f"({stats['prune_ratio']*100:.2f}%)")
    print(f"  N_keep           = {stats['N_keep']}")
    if "overflow_p50" in stats:
        print(f"  overflow (可见 minview): "
              f"p50={stats['overflow_p50']:.3f} "
              f"p90={stats['overflow_p90']:.3f} "
              f"p99={stats['overflow_p99']:.3f}")
    print("=" * 60)

    if args.dry_run:
        print("[dry_run] 不保存, 结束.")
        return

    # 执行剪枝并保存
    # 注意: 加载模式下 gaussians.optimizer 是 None, 不能走 gaussians.prune_points
    # (它内部会调 _prune_optimizer 依赖 optimizer.param_groups)
    # 因此直接对 6 个核心 tensor 做布尔筛选, 保留掩码取反 (True=保留)
    if stats["N_prune"] > 0:
        keep = ~prune_mask
        with torch.no_grad():
            gaussians._xyz          = torch.nn.Parameter(gaussians._xyz[keep].requires_grad_(True))
            gaussians._features_dc  = torch.nn.Parameter(gaussians._features_dc[keep].requires_grad_(True))
            gaussians._features_rest = torch.nn.Parameter(gaussians._features_rest[keep].requires_grad_(True))
            gaussians._scaling      = torch.nn.Parameter(gaussians._scaling[keep].requires_grad_(True))
            gaussians._rotation     = torch.nn.Parameter(gaussians._rotation[keep].requires_grad_(True))
            gaussians._opacity      = torch.nn.Parameter(gaussians._opacity[keep].requires_grad_(True))
            if hasattr(gaussians, "max_radii2D") and gaussians.max_radii2D.numel() > 0:
                gaussians.max_radii2D = gaussians.max_radii2D[keep]

    # 保存到新 iteration 目录
    out_dir = os.path.join(dataset.model_path, "point_cloud",
                           f"iteration_{args.output_iter}")
    os.makedirs(out_dir, exist_ok=True)
    ply_path = os.path.join(out_dir, "point_cloud.ply")
    gaussians.save_ply(ply_path)
    print(f"[offline-prune] saved pruned model: {ply_path}")
    print(f"[offline-prune] N after prune: {gaussians.get_xyz.shape[0]}")
    print(f"\n下一步 render:")
    print(f"  python render.py -m {dataset.model_path} "
          f"--eval --iteration {args.output_iter}")
    print(f"  python metrics.py -m {dataset.model_path}")


if __name__ == "__main__":
    main()
