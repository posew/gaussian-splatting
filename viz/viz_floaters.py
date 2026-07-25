"""
viz_floaters.py - 漂浮物诊断可视化

对应 IntentSplat_Plan.md M0.3

支持:
  1. xz 俯视图 (topdown): .ply 载入所有高斯 xyz + opacity, 按 avg_wm > thr 分红/蓝
     - 红 = 前景 (fg, avg_wm > thr, 落在意图区)
     - 蓝 = 背景 (bg, 该被剔的雾状高斯候选)
  2. 三联图 (triptych, per-view):
     - 左: render (从 test set 目录直接读)
     - 中: |render - gt| 误差热图
     - 右: render 上叠加 bg 高斯投影 (蓝色小点)

用法:
    # 只出 xz 俯视图 (最快, 不需要 render)
    python viz/viz_floaters.py -m <model_path> --iteration 30000 --topdown_only

    # 出 xz 俯视图 + 每个 test view 的三联图 (需要 wm)
    python viz/viz_floaters.py -m <model_path> --iteration 30000
"""

import os
import sys
import argparse
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

# 允许从 gaussian-splatting 根目录跑
GS_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if GS_ROOT not in sys.path:
    sys.path.insert(0, GS_ROOT)


# ─────────────────────────────────────────────
# 工具: 从 ply 直接读 xyz + opacity + scale (不用 GaussianModel, 避免 CUDA 依赖)
# ─────────────────────────────────────────────

def load_ply_lite(ply_path: str):
    """
    读 3DGS 输出的 .ply, 返回 dict:
        xyz:      (N, 3) float32
        opacity:  (N,) float32 (raw, 经过 sigmoid 前的; 用户自行 sigmoid)
        scale:    (N, 3) float32 (log-scale, 用户自行 exp)
    """
    from plyfile import PlyData
    ply = PlyData.read(ply_path)
    v = ply["vertex"]
    xyz = np.stack([np.asarray(v["x"]), np.asarray(v["y"]), np.asarray(v["z"])], axis=1).astype(np.float32)
    opacity = np.asarray(v["opacity"], dtype=np.float32)
    scale = np.stack([np.asarray(v[f"scale_{i}"]) for i in range(3)], axis=1).astype(np.float32)
    return {"xyz": xyz, "opacity_raw": opacity, "scale_raw": scale}


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


# ─────────────────────────────────────────────
# 工具: 加载 wm 目录 + cameras.json (拿相机外参投影 avg_wm)
# ─────────────────────────────────────────────

def load_cameras_json(cameras_json: str):
    """
    读 3DGS 训练时 dump 的 cameras.json.
    返回 list of dict, 每项:
        {name, id, fx, fy, width, height, R, T (world-to-cam), position}
    """
    import json
    with open(cameras_json, "r") as f:
        cams = json.load(f)
    out = []
    for c in cams:
        R = np.asarray(c["rotation"], dtype=np.float32)  # (3,3), world -> cam
        T = np.asarray(c["position"], dtype=np.float32)  # (3,), cam center in world
        out.append({
            "name": c["img_name"],
            "id": c["id"],
            "fx": c["fx"], "fy": c["fy"],
            "width": c["width"], "height": c["height"],
            "R": R, "position": T,
        })
    return out


def load_wm(wm_dir: str, name: str, target_hw: tuple):
    """加载 wm 到 numpy (H, W) float32, 未找到返 None."""
    if wm_dir is None or not os.path.isdir(wm_dir):
        return None
    stem = os.path.splitext(name)[0]
    for ext in [".npy"]:
        p = os.path.join(wm_dir, stem + ext)
        if os.path.exists(p):
            arr = np.load(p).astype(np.float32)
            return _resize_np(arr, target_hw)
    for ext in [".png", ".jpg", ".jpeg"]:
        p = os.path.join(wm_dir, stem + ext)
        if os.path.exists(p):
            im = np.array(Image.open(p).convert("L"), dtype=np.float32) / 255.0
            return _resize_np(im, target_hw)
    return None


def _resize_np(arr, target_hw):
    import cv2
    H, W = target_hw
    if arr.ndim == 3:
        arr = arr[..., 0]
    if arr.shape[0] != H or arr.shape[1] != W:
        arr = cv2.resize(arr, (W, H), interpolation=cv2.INTER_LINEAR)
    return arr.astype(np.float32)


# ─────────────────────────────────────────────
# 计算每颗高斯的 avg_wm (numpy, 无需 CUDA)
# ─────────────────────────────────────────────

def compute_avg_wm_per_gaussian(xyz: np.ndarray, cameras: list, wm_dir: str,
                                 wm_thr: float = 0.5):
    """
    对每颗高斯: 投影到每个视角 -> 若在图像内, 取该像素的 wm -> 平均.
    返回:
        avg_wm:  (N,) float32, 未见过时 = nan
        n_seen:  (N,) int32
    """
    N = xyz.shape[0]
    sum_wm = np.zeros(N, dtype=np.float32)
    n_seen = np.zeros(N, dtype=np.int32)

    for cam in cameras:
        H, W = int(cam["height"]), int(cam["width"])
        wm = load_wm(wm_dir, cam["name"], (H, W))
        if wm is None:
            continue

        # world -> cam. 注意 cameras.json 里 rotation 是 c2w 还是 w2c 需要确认.
        # 从 scene.dataset_readers 看, dump 时 R.T = R_w2c (是 T 表示 cam center in world)
        # 保险起见: 用 R 直接测. 如果结果反着, 用 R.T 再试.
        R = cam["R"]  # 假设是 w2c (or c2w?), 下文都用同一约定
        cam_center = cam["position"]

        # 标准 3DGS cameras.json 里 rotation 是 R_w2c (与训练时一致), position 是 cam center
        # world-to-cam: p_cam = R @ (p_world - cam_center)? 还是 p_cam = R @ p_world + t?
        # 常见约定: cameras.json 保存的是 c2w, 即 R@p_cam + t = p_world, 所以 w2c: R^T @ (p_world - t)
        # 我们按 c2w 处理:
        rel = xyz - cam_center[None, :]  # (N, 3)
        cam_xyz = rel @ R  # (N, 3)  == (R^T @ rel^T)^T = rel @ R (若 R 正交)
        depth = cam_xyz[:, 2]
        valid = depth > 0.01

        # 简单 pinhole 投影
        fx, fy = cam["fx"], cam["fy"]
        cx, cy = W / 2.0, H / 2.0
        u = fx * cam_xyz[:, 0] / (depth + 1e-8) + cx
        v = fy * cam_xyz[:, 1] / (depth + 1e-8) + cy
        in_img = valid & (u >= 0) & (u < W) & (v >= 0) & (v < H)

        idx = np.where(in_img)[0]
        if len(idx) == 0:
            continue
        ui = np.clip(u[idx].astype(np.int32), 0, W - 1)
        vi = np.clip(v[idx].astype(np.int32), 0, H - 1)
        sum_wm[idx] += wm[vi, ui]
        n_seen[idx] += 1

    avg = np.where(n_seen > 0, sum_wm / np.maximum(n_seen, 1), np.nan)
    return avg, n_seen


# ─────────────────────────────────────────────
# 主视图: xz 俯视
# ─────────────────────────────────────────────

def plot_xz_topdown(xyz: np.ndarray, avg_wm: np.ndarray, opacity: np.ndarray,
                    wm_thr: float, out_path: str,
                    opacity_thr: float = 0.1, subsample: int = 1):
    """
    xz 俯视: x -> 横轴, z -> 纵轴 (相机深度).
    - 高 opacity + avg_wm > thr = 红 (前景)
    - 高 opacity + avg_wm < thr = 蓝 (bg 漂浮物候选)
    - avg_wm 为 nan (未见过) = 灰
    """
    if subsample > 1:
        xyz = xyz[::subsample]
        avg_wm = avg_wm[::subsample]
        opacity = opacity[::subsample]

    opaque = opacity > opacity_thr
    seen = ~np.isnan(avg_wm)
    fg = opaque & seen & (avg_wm > wm_thr)
    bg = opaque & seen & (avg_wm <= wm_thr)
    unseen = opaque & (~seen)

    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    if unseen.any():
        ax.scatter(xyz[unseen, 0], xyz[unseen, 2], s=0.3, c="gray", alpha=0.2, label=f"unseen ({unseen.sum()})")
    if bg.any():
        ax.scatter(xyz[bg, 0], xyz[bg, 2], s=0.6, c="tab:blue", alpha=0.6, label=f"bg-floater ({bg.sum()})")
    if fg.any():
        ax.scatter(xyz[fg, 0], xyz[fg, 2], s=0.5, c="tab:red", alpha=0.6, label=f"fg ({fg.sum()})")
    ax.set_xlabel("x (world)")
    ax.set_ylabel("z (world)")
    ax.set_title(f"xz top-down (opacity>{opacity_thr}, wm_thr={wm_thr})")
    ax.set_aspect("equal", adjustable="datalim")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    print(f"  [xz_topdown] saved -> {out_path}")
    return {
        "n_fg": int(fg.sum()), "n_bg": int(bg.sum()),
        "n_unseen_opaque": int(unseen.sum()),
        "ratio_bg_over_fg": float(bg.sum() / max(fg.sum(), 1)),
    }


# ─────────────────────────────────────────────
# 三联图 (per view)
# ─────────────────────────────────────────────

def plot_triptych(render_path: str, gt_path: str, wm: np.ndarray, out_path: str,
                  wm_thr: float = 0.5):
    import cv2
    r = np.asarray(Image.open(render_path).convert("RGB"), dtype=np.float32) / 255.0
    g = np.asarray(Image.open(gt_path).convert("RGB"), dtype=np.float32) / 255.0
    H, W = r.shape[:2]
    if wm is not None:
        wm_r = _resize_np(wm, (H, W))
        mask = (wm_r > wm_thr).astype(np.float32)
    else:
        wm_r = np.ones((H, W), dtype=np.float32)
        mask = wm_r

    err = np.abs(r - g).mean(axis=2)  # (H, W)

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    axes[0].imshow(r)
    axes[0].set_title("render")
    axes[0].axis("off")
    axes[1].imshow(err, cmap="hot", vmin=0, vmax=0.5)
    axes[1].set_title(f"|render - gt| (mean={err.mean():.4f})")
    axes[1].axis("off")

    # 第三张: render 上把 bg 区染蓝 (示意 bg 有多雾)
    bg = (mask < 0.5).astype(np.float32)
    r_bg_hi = r.copy()
    r_bg_hi[..., 2] = np.clip(r_bg_hi[..., 2] + bg * 0.3, 0, 1)  # bg 区加蓝
    axes[2].imshow(r_bg_hi)
    axes[2].set_title(f"render, bg overlay blue (bg_cov={bg.mean():.2%})")
    axes[2].axis("off")
    fig.tight_layout()
    fig.savefig(out_path, dpi=100)
    plt.close(fig)


# ─────────────────────────────────────────────
# main
# ─────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-m", "--model_path", required=True)
    ap.add_argument("--iteration", type=int, default=30000)
    ap.add_argument("--wm_dir", default=None,
                    help="weight_maps 目录; 不给则从 cfg_args 的 source_path/weight_maps 找")
    ap.add_argument("--wm_thr", type=float, default=0.5)
    ap.add_argument("--opacity_thr", type=float, default=0.1)
    ap.add_argument("--subsample", type=int, default=1,
                    help="topdown 图散点抽样, 大场景 (>500k) 可用 5-10 加速")
    ap.add_argument("--topdown_only", action="store_true",
                    help="只出 xz 俯视图, 不出 per-view 三联图")
    ap.add_argument("--triptych_max", type=int, default=5,
                    help="最多输出多少张 per-view 三联图")
    ap.add_argument("--out_dir", default=None,
                    help="输出目录 (默认 <model_path>/viz_floaters_iter<N>)")
    args = ap.parse_args()

    mp = Path(args.model_path)
    if args.out_dir is None:
        out_dir = mp / f"viz_floaters_iter{args.iteration}"
    else:
        out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    ply_path = mp / f"point_cloud/iteration_{args.iteration}/point_cloud.ply"
    if not ply_path.exists():
        raise FileNotFoundError(f"ply not found: {ply_path}")

    # ── wm_dir 解析 ──
    wm_dir = args.wm_dir
    if wm_dir is None:
        cfg = mp / "cfg_args"
        if cfg.exists():
            import re
            m = re.search(r"source_path=['\"]([^'\"]+)['\"]", cfg.read_text())
            if m:
                candidate = Path(m.group(1)) / "weight_maps"
                if candidate.is_dir():
                    wm_dir = str(candidate)
    print(f"model_path = {args.model_path}")
    print(f"ply = {ply_path}")
    print(f"wm_dir = {wm_dir}")
    print(f"out_dir = {out_dir}")

    # ── 载 ply ──
    print("loading ply...")
    ply = load_ply_lite(str(ply_path))
    xyz = ply["xyz"]
    opacity = sigmoid(ply["opacity_raw"])
    print(f"  N = {len(xyz)}, opacity>{args.opacity_thr}: {(opacity > args.opacity_thr).sum()}")

    # ── 计算 avg_wm ──
    cameras_json = mp / "cameras.json"
    if wm_dir is not None and cameras_json.exists():
        print("loading cameras.json + wm to compute avg_wm...")
        cams = load_cameras_json(str(cameras_json))
        print(f"  {len(cams)} cameras")
        avg_wm, n_seen = compute_avg_wm_per_gaussian(xyz, cams, wm_dir, wm_thr=args.wm_thr)
        print(f"  n_seen>0: {(n_seen > 0).sum()}")
    else:
        avg_wm = np.full(len(xyz), np.nan, dtype=np.float32)
        n_seen = np.zeros(len(xyz), dtype=np.int32)
        print("  [warn] cameras.json 或 wm_dir 未找到, avg_wm 全 nan (只能出灰色 topdown)")

    # ── xz 俯视 ──
    xz_out = out_dir / f"xz_topdown_iter{args.iteration}.png"
    stats = plot_xz_topdown(xyz, avg_wm, opacity, wm_thr=args.wm_thr,
                            out_path=str(xz_out),
                            opacity_thr=args.opacity_thr, subsample=args.subsample)
    print(f"  xz_topdown stats: {stats}")

    # ── 三联图 (若给了 render dir) ──
    if not args.topdown_only:
        test_dir = mp / f"test/ours_{args.iteration}"
        renders = test_dir / "renders"
        gts = test_dir / "gt"
        if renders.is_dir() and gts.is_dir():
            fnames = sorted(os.listdir(renders))[: args.triptych_max]
            for fn in fnames:
                stem = os.path.splitext(fn)[0]
                # 尝试对应 wm (train view 与 test view 命名不同, 常见 test 用 idx 递增, 没直接对应 wm)
                # 兜底: 若找不到 wm 就画 render + err, 不上 bg overlay
                r_img = Image.open(renders / fn)
                W, H = r_img.size
                wm = None
                if wm_dir is not None:
                    # 试若干可能的命名
                    for cand in [stem, stem.lstrip("0"), stem.zfill(4)]:
                        wm = load_wm(wm_dir, cand, (H, W))
                        if wm is not None:
                            break
                trip_out = out_dir / f"triptych_{stem}.png"
                plot_triptych(str(renders / fn), str(gts / fn), wm, str(trip_out),
                              wm_thr=args.wm_thr)
                print(f"  [triptych] {fn} -> {trip_out} (wm found: {wm is not None})")
        else:
            print(f"  [skip triptych] {test_dir} 下没找到 renders/gt, 请先 render.py")

    # ── 汇总 json ──
    import json
    summary = {
        "model_path": args.model_path, "iteration": args.iteration,
        "wm_thr": args.wm_thr, "opacity_thr": args.opacity_thr,
        "topdown_stats": stats,
        "n_ply_total": int(len(xyz)),
        "n_opaque": int((opacity > args.opacity_thr).sum()),
        "n_seen_any_view": int((n_seen > 0).sum()),
    }
    with open(out_dir / "viz_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"summary -> {out_dir / 'viz_summary.json'}")


if __name__ == "__main__":
    main()
