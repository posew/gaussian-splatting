"""
dump_weight_map_v2.py — 对指定 source_path 里的若干采样图, 输出
一张四联对比图:  raw | legacy(locvar_dist) | kmeans_ab | +caustic | wm_v2(+colmap_seed)

对应 IntentSplat_Plan.md M1.4.

用法示例:
    python scripts/dump_weight_map_v2.py \
        --source_path /data3/ycf/Data/underwaterMachine \
        --out_dir /data1/ycf/academic/github_run/RUN_RESULTS/wm_v2_dump_2026-07-25 \
        --n_samples 5

若 --image_stems 显式指定 (逗号分隔), 则忽略 --n_samples.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import cv2
import numpy as np

# 让脚本能以 `python scripts/dump_weight_map_v2.py` 运行, 也支持在
# gaussian-splatting/ 根目录下直接 -m scripts.dump_weight_map_v2.
_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from utils.weight_map_utils import (  # noqa: E402
    compute_kmeans_ab_weight_map,
    compute_caustic_mask,
    compute_wm_v2,
    compute_locvar_dist_weight_map,
)
from utils.colmap_seeds import build_colmap_seed_masks  # noqa: E402


# ─────────────────────────────────────────────
# 可视化工具
# ─────────────────────────────────────────────
def _norm01_to_bgr(x: np.ndarray) -> np.ndarray:
    """把 [0,1] float 图上色 (JET), 输出 uint8 BGR."""
    x = np.clip(x, 0.0, 1.0)
    x8 = (x * 255).astype(np.uint8)
    return cv2.applyColorMap(x8, cv2.COLORMAP_JET)


def _overlay(img_bgr: np.ndarray, wm: np.ndarray, alpha: float = 0.55) -> np.ndarray:
    """把 wm 半透叠加到原图上, 便于人眼看 mask 打在什么位置."""
    heat = _norm01_to_bgr(wm)
    return cv2.addWeighted(img_bgr, 1.0 - alpha, heat, alpha, 0.0)


def _put_label(img: np.ndarray, text: str) -> np.ndarray:
    """左上角画一条标题."""
    out = img.copy()
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
    cv2.rectangle(out, (0, 0), (tw + 14, th + 14), (0, 0, 0), -1)
    cv2.putText(out, text, (7, th + 7),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
    return out


def _hstack_pad(imgs: list[np.ndarray], pad: int = 4,
                pad_color: tuple = (32, 32, 32)) -> np.ndarray:
    """把一排图横向拼在一起, 中间加一条分隔线."""
    if not imgs:
        return np.zeros((10, 10, 3), np.uint8)
    h = max(im.shape[0] for im in imgs)
    resized = []
    for im in imgs:
        if im.shape[0] != h:
            r = h / float(im.shape[0])
            im = cv2.resize(im, (int(round(im.shape[1] * r)), h))
        resized.append(im)
    sep = np.full((h, pad, 3), pad_color, np.uint8)
    out = []
    for i, im in enumerate(resized):
        out.append(im)
        if i != len(resized) - 1:
            out.append(sep)
    return np.hstack(out)


# ─────────────────────────────────────────────
# 采样图选取
# ─────────────────────────────────────────────
def _list_images(source_path: str) -> list[str]:
    img_dir = os.path.join(source_path, "images")
    if not os.path.isdir(img_dir):
        raise FileNotFoundError(f"没找到 {img_dir}")
    files = sorted([f for f in os.listdir(img_dir)
                    if f.lower().endswith((".jpg", ".jpeg", ".png"))])
    return [os.path.join(img_dir, f) for f in files]


def _pick_samples(all_paths: list[str], stems: list[str] | None,
                  n: int) -> list[str]:
    if stems:
        stem_set = {s.strip() for s in stems if s.strip()}
        picked = [p for p in all_paths
                  if os.path.splitext(os.path.basename(p))[0] in stem_set]
        missing = stem_set - {os.path.splitext(os.path.basename(p))[0]
                              for p in picked}
        if missing:
            print(f"[warn] 未找到这些 stem: {sorted(missing)}")
        return picked
    if len(all_paths) <= n:
        return all_paths
    # 均匀取 n 张
    idx = np.linspace(0, len(all_paths) - 1, num=n, dtype=int)
    return [all_paths[i] for i in idx]


# ─────────────────────────────────────────────
# 主流程
# ─────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source_path", required=True,
                    help="COLMAP scene 根 (含 images/ 与 sparse/0/)")
    ap.add_argument("--out_dir", required=True, help="对比图输出目录")
    ap.add_argument("--n_samples", type=int, default=5,
                    help="均匀采样几张图; 若指定 --image_stems 则忽略")
    ap.add_argument("--image_stems", type=str, default="",
                    help="指定要 dump 的图 stem, 逗号分隔 (例如 0001,0050,0100)")
    ap.add_argument("--kmeans_k", type=int, default=16)
    ap.add_argument("--caustic_L", type=int, default=200)
    ap.add_argument("--caustic_chroma", type=int, default=40)
    ap.add_argument("--caustic_dilate", type=int, default=9)
    ap.add_argument("--seed_radius", type=int, default=15)
    ap.add_argument("--no_seed", action="store_true",
                    help="跳过 COLMAP seed 构建 (纯 kmeans_ab + caustic 对比)")
    ap.add_argument("--dump_caustic_alone", action="store_true",
                    help="额外输出一张 caustic mask 的独立 png")
    ap.add_argument("--dump_all_wm", action="store_true",
                    help="对所有图批量 dump wm_v2 到 <out_dir>/*.npy (给 metrics.py 用)")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    all_paths = _list_images(args.source_path)
    stems = args.image_stems.split(",") if args.image_stems else None
    if args.dump_all_wm:
        samples = all_paths
        print(f"[dump_wm_v2] dump_all_wm 模式: {len(samples)} 张全量")
    else:
        samples = _pick_samples(all_paths, stems, args.n_samples)
        if not samples:
            print("[err] 没有采样到任何图, 退出.")
            return
        print(f"[dump_wm_v2] 采样 {len(samples)} 张: "
              + ", ".join(os.path.basename(p) for p in samples))

    # 构建 colmap seeds (若开启), 按第一张图的分辨率作为 target_hw
    seeds = {}
    if not args.no_seed:
        first = cv2.imread(samples[0], cv2.IMREAD_COLOR)
        H0, W0 = first.shape[:2]
        try:
            seeds = build_colmap_seed_masks(
                args.source_path, target_hw=(H0, W0),
                radius_px=args.seed_radius,
            )
            print(f"[dump_wm_v2] colmap_seeds ready: {len(seeds)} images @ ({H0},{W0})")
        except Exception as e:
            print(f"[warn] 构建 colmap_seeds 失败, 后面回退到无 seed: {e}")
            seeds = {}

    for path in samples:
        img = cv2.imread(path, cv2.IMREAD_COLOR)
        if img is None:
            print(f"[skip] 读不到 {path}")
            continue
        stem = os.path.splitext(os.path.basename(path))[0]
        H, W = img.shape[:2]

        # legacy: locvar_dist (M0 baseline 用的)
        try:
            wm_legacy = compute_locvar_dist_weight_map(img)
        except Exception as e:
            print(f"[warn] legacy locvar_dist 失败({stem}): {e}, 用零图代替")
            wm_legacy = np.zeros((H, W), np.float32)

        # M1.1: kmeans_ab
        wm_km = compute_kmeans_ab_weight_map(img, k=args.kmeans_k)

        # M1.2: caustic mask + km * (1-caustic)
        caustic = compute_caustic_mask(
            img, thr_L=args.caustic_L, thr_chroma=args.caustic_chroma,
            dilate=args.caustic_dilate,
        )
        wm_km_nocaustic = wm_km * (1.0 - caustic)

        # M1.3 + M1.wm_v2: fuse kmeans_ab, caustic, colmap_seed
        seed_mask = None
        if stem in seeds and seeds[stem].size:
            m = seeds[stem]
            if m.shape != (H, W):
                m = cv2.resize(m, (W, H), interpolation=cv2.INTER_NEAREST)
            seed_mask = m
        v2 = compute_wm_v2(
            img, colmap_seed_mask=seed_mask,
            k=args.kmeans_k,
            caustic_thr_L=args.caustic_L,
            caustic_thr_chroma=args.caustic_chroma,
            caustic_dilate=args.caustic_dilate,
        )
        wm_final = v2["wm"]

        # dump_all_wm 模式: 只存 npy, 跳过可视化
        if args.dump_all_wm:
            np.save(os.path.join(args.out_dir, f"{stem}.npy"),
                    wm_final.astype(np.float32))
            if (len(samples) < 20) or (samples.index(path) % 50 == 0):
                print(f"  [{stem}] wm_mean={float(wm_final.mean()):.3f} "
                      f"caustic_cov={float(caustic.mean()):.3%} "
                      f"seed_cov={float(seed_mask.mean()) if seed_mask is not None else 0:.3%}")
            continue

        # 构图
        pane_raw = _put_label(img, f"[{stem}] raw")
        pane_legacy = _put_label(_overlay(img, wm_legacy), "legacy(locvar_dist)")
        pane_km = _put_label(_overlay(img, wm_km), "kmeans_ab")
        pane_km_nc = _put_label(_overlay(img, wm_km_nocaustic), "kmeans_ab * (1-caustic)")
        if seed_mask is not None:
            v2_label = "wm_v2 (+colmap_seed)"
        else:
            v2_label = "wm_v2 (no seed)"
        pane_v2 = _put_label(_overlay(img, wm_final), v2_label)

        row = _hstack_pad([pane_raw, pane_legacy, pane_km, pane_km_nc, pane_v2])
        out_path = os.path.join(args.out_dir, f"{stem}_wm_compare.jpg")
        cv2.imwrite(out_path, row, [cv2.IMWRITE_JPEG_QUALITY, 88])

        # 额外: 单独 dump caustic mask
        if args.dump_caustic_alone:
            cv2.imwrite(
                os.path.join(args.out_dir, f"{stem}_caustic.png"),
                (caustic * 255).astype(np.uint8),
            )
            if seed_mask is not None:
                cv2.imwrite(
                    os.path.join(args.out_dir, f"{stem}_colmap_seed.png"),
                    (seed_mask * 255).astype(np.uint8),
                )

        # 简报
        km_mean = float(wm_km.mean())
        km_nc_mean = float(wm_km_nocaustic.mean())
        v2_mean = float(wm_final.mean())
        caustic_cov = float(caustic.mean())
        seed_cov = float(seed_mask.mean()) if seed_mask is not None else 0.0
        print(f"  [{stem}] km_mean={km_mean:.3f} km_nocaustic_mean={km_nc_mean:.3f} "
              f"v2_mean={v2_mean:.3f} caustic_cov={caustic_cov:.3%} "
              f"seed_cov={seed_cov:.3%}")

    print(f"[dump_wm_v2] done -> {args.out_dir}")


if __name__ == "__main__":
    main()
