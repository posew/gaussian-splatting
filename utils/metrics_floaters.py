"""
metrics_floaters.py - 漂浮物 (floaters) 量化指标

对应 IntentSplat_Plan.md M0.2

三个指标 (至少任选两个即可):
  A. bg 区渲染 depth 方差 (退化版, 因为暂无单目 depth)
     原始计划: 渲染 depth vs 单目 depth L1
     退化理由: 单目 depth 属于 M6 范畴, 本 milestone 先用 bg 区 depth 稳定性
              代理 - 若 bg 区都是漂浮物, depth 会跨像素跳动
  B. bg 视锥内 opacity>0.1 的高斯数
     - 每帧把所有高斯投影到 image plane, 落在 bg mask (wm<thr) 内且
       opacity>0.1 且 depth>0 (前方) 的计数, 跨帧累计
  C. bg 区 alpha 累积到 0.5 的深度差
     - 现有 diff_gaussian_rasterization 不返回 per-pixel alpha 层, 跳过
     - 若之后接入 3DGS-Depth extension 可补上

用法 (从 python 调用):
    from utils.metrics_floaters import (
        bg_depth_variance, count_bg_opaque_gaussians, floater_score_full
    )
    score = floater_score_full(gaussians, cameras, wm_dict, thr=0.5, pipe=...)
    -> {"bg_depth_var": 0.045, "bg_gaussian_count": 12345, ...}
"""

import numpy as np
import torch
import torch.nn.functional as F


# ─────────────────────────────────────────────
# 指标 A: bg 区渲染 depth 方差
# ─────────────────────────────────────────────

def bg_depth_variance(depth_map: torch.Tensor, wm: torch.Tensor,
                      thr: float = 0.5) -> float:
    """
    输入:
        depth_map: (1, H, W) 或 (H, W), 渲染 depth
        wm:        (1, H, W), weight map, 与 depth 同分辨率
        thr:       fg/bg 阈值
    输出:
        bg 区 depth 的方差 (float). 若 bg 面积过小返回 nan.

    直观: bg 区都是水 -> 期望 depth 空/均匀; 若充满漂浮物 -> depth 高频跳动.
    """
    d = depth_map
    if d.dim() == 3:
        d = d[0]
    m = wm
    if m.dim() == 3:
        m = m[0]
    bg = (m <= thr) & (d > 0)
    if bg.sum().item() < 100:
        return float("nan")
    d_bg = d[bg]
    return float(d_bg.var().item())


def bg_depth_gradient_mag(depth_map: torch.Tensor, wm: torch.Tensor,
                          thr: float = 0.5) -> float:
    """
    bg 区 depth 梯度的平均模长 (漂浮物导致的高频抖动指标).
    """
    d = depth_map
    if d.dim() == 2:
        d = d.unsqueeze(0)
    if d.dim() == 3:
        d = d.unsqueeze(0)  # (1,1,H,W)
    # sobel
    kx = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=d.dtype, device=d.device)
    ky = kx.t().contiguous()
    kx = kx.view(1, 1, 3, 3)
    ky = ky.view(1, 1, 3, 3)
    gx = F.conv2d(d, kx, padding=1)
    gy = F.conv2d(d, ky, padding=1)
    grad = torch.sqrt(gx ** 2 + gy ** 2 + 1e-12)[0, 0]
    m = wm
    if m.dim() == 3:
        m = m[0]
    bg = (m <= thr) & (d[0, 0] > 0)
    if bg.sum().item() < 100:
        return float("nan")
    return float(grad[bg].mean().item())


# ─────────────────────────────────────────────
# 指标 B: bg 视锥内 opacity>0.1 的高斯数
# ─────────────────────────────────────────────

@torch.no_grad()
def _project_xyz_to_view(xyz: torch.Tensor, view_transform: torch.Tensor,
                         full_proj: torch.Tensor, H: int, W: int):
    """
    把 N 个 3D 点 (world) 投到当前视角 image plane.

    返回:
        u, v: (N,) float pixel 坐标 (可能超出 [0,W)/[0,H))
        depth_cam: (N,) float, camera-space z (正值 = 前方)
        in_frustum: (N,) bool, 是否在视锥内 (depth>0 且投影在图像内)
    """
    N = xyz.shape[0]
    ones = torch.ones((N, 1), device=xyz.device, dtype=xyz.dtype)
    xyz_h = torch.cat([xyz, ones], dim=1)  # (N, 4)

    # camera space
    cam = xyz_h @ view_transform  # (N, 4)
    depth_cam = cam[:, 2]

    # NDC / clip space
    clip = xyz_h @ full_proj
    w = clip[:, 3]
    valid_w = w.abs() > 1e-6
    ndc = clip / (w.unsqueeze(1) + 1e-8)
    x_ndc = ndc[:, 0]
    y_ndc = ndc[:, 1]

    # image space (opengl-ish: y flip)
    u = (x_ndc * 0.5 + 0.5) * W
    v = (1.0 - (y_ndc * 0.5 + 0.5)) * H  # (y 反向)

    in_frustum = valid_w & (depth_cam > 0.01) & \
                 (u >= 0) & (u < W) & (v >= 0) & (v < H)
    return u, v, depth_cam, in_frustum


@torch.no_grad()
def count_bg_opaque_gaussians(gaussians, cameras, wm_lookup: dict,
                              opacity_thr: float = 0.1, wm_thr: float = 0.5) -> dict:
    """
    统计每颗高斯是否在任何视角上都落在 bg 区.

    Args:
        gaussians:  GaussianModel, 需要 get_xyz / get_opacity
        cameras:    list of Camera, 需要 world_view_transform / full_proj_transform /
                    image_width / image_height / image_name
        wm_lookup:  {image_name: torch.Tensor (1, H, W) or numpy (H,W)}
        opacity_thr, wm_thr: 阈值

    Returns:
        {
          "n_total_opaque": N,               # opacity > opacity_thr 的所有高斯
          "n_seen_in_any_view": M,           # 至少被 1 视角看见 的
          "n_always_bg": K,                  # 见过它的所有视角里, 全落在 bg 区
          "ratio_always_bg": K / M,          # 占比
          "n_bg_hits_total": T,              # 所有视角 bg 命中的累计 (可对比 fg 累计)
        }

    直观: `n_always_bg` 就是 "该被剔的雾状高斯" 的估计.
    """
    xyz = gaussians.get_xyz.detach()  # (N, 3)
    opacity = gaussians.get_opacity.detach().squeeze(-1)  # (N,)
    N = xyz.shape[0]
    opaque_mask = opacity > opacity_thr

    seen_count = torch.zeros(N, dtype=torch.int32, device=xyz.device)
    bg_count = torch.zeros(N, dtype=torch.int32, device=xyz.device)

    for cam in cameras:
        wm_t = wm_lookup.get(cam.image_name, None)
        if wm_t is None:
            continue
        if not torch.is_tensor(wm_t):
            wm_t = torch.from_numpy(np.asarray(wm_t, dtype=np.float32))
        wm_t = wm_t.to(xyz.device)
        if wm_t.dim() == 3:
            wm_t = wm_t[0]  # (H, W)

        H, W = int(cam.image_height), int(cam.image_width)
        # resize wm 到相机分辨率 (通常一致)
        if wm_t.shape[0] != H or wm_t.shape[1] != W:
            wm_t = F.interpolate(wm_t.unsqueeze(0).unsqueeze(0),
                                 size=(H, W), mode="bilinear",
                                 align_corners=False).squeeze(0).squeeze(0)

        u, v, dz, in_fru = _project_xyz_to_view(
            xyz, cam.world_view_transform, cam.full_proj_transform, H, W)

        ui = u.long().clamp(0, W - 1)
        vi = v.long().clamp(0, H - 1)
        wm_val = wm_t[vi, ui]  # (N,)
        is_bg = (wm_val <= wm_thr)

        active = in_fru & opaque_mask
        seen_count = seen_count + active.int()
        bg_count = bg_count + (active & is_bg).int()

    seen_any = seen_count > 0
    always_bg = seen_any & (seen_count == bg_count)

    return {
        "n_total": int(N),
        "n_total_opaque": int(opaque_mask.sum().item()),
        "n_seen_in_any_view": int(seen_any.sum().item()),
        "n_always_bg": int(always_bg.sum().item()),
        "ratio_always_bg": float(always_bg.sum().item() / max(seen_any.sum().item(), 1)),
        "n_bg_hits_total": int(bg_count.sum().item()),
        "n_seen_hits_total": int(seen_count.sum().item()),
        "opacity_thr": opacity_thr,
        "wm_thr": wm_thr,
    }


# ─────────────────────────────────────────────
# 综合入口: floater_score_full
# ─────────────────────────────────────────────

@torch.no_grad()
def floater_score_full(gaussians, cameras, wm_lookup: dict,
                       render_fn, pipe, background,
                       wm_thr: float = 0.5, opacity_thr: float = 0.1) -> dict:
    """
    综合评估. 需要能 render (用于指标 A).

    Args:
        render_fn:  callable(cam, gaussians, pipe, background, ...) -> dict with "depth"
                    (即 gaussian_renderer.render)

    Returns:
        {
          "bg_depth_var_mean": ...,
          "bg_depth_grad_mean": ...,
          "gaussian_stats": {...},   # count_bg_opaque_gaussians 的输出
          "per_view_depth_var": {name: v, ...}
        }
    """
    per_view_var = {}
    per_view_grad = {}
    for cam in cameras:
        wm_t = wm_lookup.get(cam.image_name, None)
        if wm_t is None:
            continue
        if not torch.is_tensor(wm_t):
            wm_t = torch.from_numpy(np.asarray(wm_t, dtype=np.float32))
        wm_t = wm_t.to(gaussians.get_xyz.device)
        out = render_fn(cam, gaussians, pipe, background)
        depth = out.get("depth", None)
        if depth is None:
            continue
        v = bg_depth_variance(depth, wm_t, thr=wm_thr)
        g = bg_depth_gradient_mag(depth, wm_t, thr=wm_thr)
        per_view_var[cam.image_name] = v
        per_view_grad[cam.image_name] = g

    def _nanmean(d):
        vs = [x for x in d.values() if x == x]
        return float(np.mean(vs)) if vs else float("nan")

    stats = count_bg_opaque_gaussians(gaussians, cameras, wm_lookup,
                                      opacity_thr=opacity_thr, wm_thr=wm_thr)
    return {
        "bg_depth_var_mean": _nanmean(per_view_var),
        "bg_depth_grad_mean": _nanmean(per_view_grad),
        "gaussian_stats": stats,
        "per_view_depth_var": per_view_var,
        "per_view_depth_grad": per_view_grad,
    }
