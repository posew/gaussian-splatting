"""
colmap_seeds.py - 从 COLMAP sparse 生成每帧的 "种子 mask"

对应 IntentSplat_Plan.md M1.3

关键洞察: COLMAP 的 `images.bin` 里, 每个 Image 已经带 `xys` (每个 2D 观测的
像素坐标) 和 `point3D_ids` (对应的 3D 点 id, <=0 表示未三角化).

  -> **不需要重新投影**! 直接用 xys[point3D_ids > 0] 就是该视角的所有稳定
     3D 观测点在图像上的像素坐标, 是天然的 "多视角一致的物体表面点".

  -> 每个种子点画半径 R 的圆, union 起来 = seed mask, 与 wm_kmeans_ab
     做 elementwise maximum 融合, 可以补足弱纹理表面的漏检 (铁皮平面上有
     稳定 feature match 但 kmeans 分不出颜色).

这个思路与 `weight_map_utils.compute_sfm_kde_weight_map` 是同源的; 区别:
  - sfm_kde: 用高斯 KDE 得到连续密度图
  - colmap_seeds: 用离散圆盘并集得到 {0,1} 二值 seed, 用来 "无条件抬升"
                  某些像素的 wm
"""

import os
import numpy as np
import cv2


def _read_sparse(source_path: str):
    """
    读 source_path/sparse/0/images.bin + cameras.bin.
    返回 (extrs_dict, intrs_dict).
    """
    from scene.colmap_loader import (
        read_extrinsics_binary, read_intrinsics_binary,
    )
    sparse_dir = os.path.join(source_path, "sparse", "0")
    images_bin = os.path.join(sparse_dir, "images.bin")
    cameras_bin = os.path.join(sparse_dir, "cameras.bin")
    if not (os.path.exists(images_bin) and os.path.exists(cameras_bin)):
        raise FileNotFoundError(
            f"COLMAP seeds 需要 {images_bin} 和 {cameras_bin}, 请先跑 COLMAP.")
    return read_extrinsics_binary(images_bin), read_intrinsics_binary(cameras_bin)


def build_seed_mask_for_image(H: int, W: int, xys: np.ndarray,
                              point3D_ids: np.ndarray,
                              radius_px: int = 15) -> np.ndarray:
    """
    单帧种子 mask.

    Args:
        H, W:         目标分辨率
        xys:          (M, 2) 该视角的 2D 观测像素 (COLMAP 原始分辨率下)
        point3D_ids:  (M,) 对应 3D 点 id; <=0 视为未三角化, 忽略
        radius_px:    每个种子的画圆半径

    注意: 这里的 xys 是 COLMAP 原始分辨率下的坐标, **需要外部先缩到 (H, W)**.
    """
    mask = np.zeros((H, W), dtype=np.uint8)
    valid = point3D_ids > 0
    if valid.sum() == 0:
        return mask
    pts = xys[valid]
    for x, y in pts:
        xi = int(round(x))
        yi = int(round(y))
        if 0 <= xi < W and 0 <= yi < H:
            cv2.circle(mask, (xi, yi), radius_px, 1, thickness=-1)
    return mask


def build_colmap_seed_masks(source_path: str, target_hw: tuple = None,
                             radius_px: int = 15) -> dict:
    """
    对 source_path/sparse/0 里所有帧构建种子 mask.

    Args:
        source_path:   COLMAP scene 根
        target_hw:     (H, W) 训练分辨率; None 时用 intrinsic 原分辨率
        radius_px:     每个种子画圆半径 (target_hw 下的像素数)

    Returns:
        {image_name_stem: uint8 (H, W)}  (stem 例如 "0001", 不含扩展名)
    """
    extrs, intrs = _read_sparse(source_path)
    out = {}
    for img_id, extr in extrs.items():
        stem = os.path.splitext(extr.name)[0]
        cam = intrs[extr.camera_id]
        H_full, W_full = int(cam.height), int(cam.width)

        if target_hw is not None:
            H_use, W_use = int(target_hw[0]), int(target_hw[1])
            sx = W_use / float(W_full)
            sy = H_use / float(H_full)
        else:
            H_use, W_use = H_full, W_full
            sx = sy = 1.0

        xys = extr.xys.copy().astype(np.float32)
        xys[:, 0] *= sx
        xys[:, 1] *= sy
        # radius 按最短边比例微调 (下采样时 seed 圆同比缩)
        r_use = max(3, int(round(radius_px * min(sx, sy))))
        mask = build_seed_mask_for_image(H_use, W_use, xys, extr.point3D_ids,
                                          radius_px=r_use)
        out[stem] = mask
    return out


# ─────────────────────────────────────────────
# 独立可执行: 预计算并保存到 source_path/colmap_seeds/
# ─────────────────────────────────────────────

def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--source_path", required=True, help="COLMAP scene 根 (含 sparse/0)")
    ap.add_argument("--out_dir", default=None,
                    help="种子 mask 输出目录, 默认 <source_path>/colmap_seeds")
    ap.add_argument("--target_h", type=int, default=None)
    ap.add_argument("--target_w", type=int, default=None)
    ap.add_argument("--radius", type=int, default=15)
    args = ap.parse_args()

    hw = None
    if args.target_h and args.target_w:
        hw = (args.target_h, args.target_w)
    masks = build_colmap_seed_masks(args.source_path, target_hw=hw,
                                    radius_px=args.radius)
    out_dir = args.out_dir or os.path.join(args.source_path, "colmap_seeds")
    os.makedirs(out_dir, exist_ok=True)
    n_pos = 0
    for stem, m in masks.items():
        cv2.imwrite(os.path.join(out_dir, stem + ".png"), m * 255)
        n_pos += int(m.sum() > 0)
    print(f"[colmap_seeds] wrote {len(masks)} masks to {out_dir} ({n_pos} non-empty)")


if __name__ == "__main__":
    main()
