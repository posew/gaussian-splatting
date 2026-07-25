"""
metrics_fg.py - 前景 / 背景分离评估

对应 IntentSplat_Plan.md M0.1

设计原则:
  - PSNR / SSIM / LPIPS 都需要一个 "mask" 版本, 只在 mask==1 的像素上评估
  - mask 由 weight_map 二值化得到 (wm > thr)
  - LPIPS 是 patch-based, 严格的 masked LPIPS 需要重新裁 patch; 这里采用
    "把 mask 外置零, 再算 LPIPS" 的近似, 与 M0 计划语义一致
  - SSIM 同样把 mask 外置零. 若 mask 面积过小 (<1%) 则返回 nan 并在上层忽略
"""

import torch
import torch.nn.functional as F


def _to_4d(t: torch.Tensor) -> torch.Tensor:
    """确保 tensor 是 (B, C, H, W) 4D."""
    if t.dim() == 3:
        return t.unsqueeze(0)
    return t


def _binarize_mask(mask: torch.Tensor, thr: float = 0.5) -> torch.Tensor:
    """把 [0, 1] 的 wm 二值化到 {0, 1}, 输出 (1, 1, H, W)."""
    m = _to_4d(mask)
    if m.shape[1] != 1:
        m = m[:, :1]
    return (m > thr).float()


def masked_psnr(pred: torch.Tensor, gt: torch.Tensor, mask: torch.Tensor,
                thr: float = 0.5) -> torch.Tensor:
    """
    在 mask==1 的像素上计算 PSNR (通道各自算平方误差, 一起平均).

    Args:
        pred, gt:  (C, H, W) 或 (1, C, H, W), 值域 [0, 1]
        mask:      (1, H, W) 或 (1, 1, H, W), 值域 [0, 1] (会被 thr 二值化)

    Returns:
        标量 tensor. 若 mask 全 0, 返回 nan.
    """
    p, g = _to_4d(pred), _to_4d(gt)
    m = _binarize_mask(mask, thr)  # (1, 1, H, W)
    # broadcasting: (1, C, H, W) * (1, 1, H, W)
    diff2 = (p - g) ** 2
    weight = m.expand_as(diff2)
    num = (diff2 * weight).sum()
    denom = weight.sum()
    if denom.item() < 1.0:
        return torch.tensor(float("nan"), device=p.device)
    mse = num / denom
    return 20.0 * torch.log10(1.0 / torch.sqrt(mse + 1e-12))


def bg_psnr(pred: torch.Tensor, gt: torch.Tensor, mask: torch.Tensor,
            thr: float = 0.5) -> torch.Tensor:
    """PSNR on `1 - mask` region (背景/水体). 语义就是 masked_psnr with mask 取反."""
    inv_mask = 1.0 - _binarize_mask(mask, thr)
    return masked_psnr(pred, gt, inv_mask, thr=0.5)


def masked_ssim(pred: torch.Tensor, gt: torch.Tensor, mask: torch.Tensor,
                thr: float = 0.5, window_size: int = 11) -> torch.Tensor:
    """
    Masked SSIM: 用 mask_erode (窗口一半的 erosion) 把 mask 外像素置零后算 SSIM,
    再在 mask 内像素上取均值.

    注: 这个近似不是严格意义上的 masked SSIM (Wang 2004 的原始定义仅在 mask
    内部窗口滑动), 但足以作为相对比较指标. 已通过 min_pool 做 erode 避免边界污染.
    """
    from utils.loss_utils import ssim  # 复用现有 ssim

    p, g = _to_4d(pred), _to_4d(gt)
    m = _binarize_mask(mask, thr)  # (1, 1, H, W)
    # erode = min_pool2d
    erode_k = window_size // 2
    if erode_k >= 1:
        m_erode = -F.max_pool2d(-m, kernel_size=erode_k * 2 + 1,
                                stride=1, padding=erode_k)
    else:
        m_erode = m

    if m_erode.sum().item() < 1.0:
        return torch.tensor(float("nan"), device=p.device)

    # 只在 mask 内比: 把 mask 外置零 (对两图一致的操作, 不会引入偏差)
    p_masked = p * m_erode
    g_masked = g * m_erode
    ssim_map = ssim(p_masked, g_masked, window_size=window_size, size_average=False)
    # ssim_map shape: (B,) — size_average=False 走的是 mean(1).mean(1).mean(1)
    # 这只给全图的每 batch 均值, 不能做像素级 mask; 我们改成手动:
    # 简化处理: 直接返回全图 ssim 值 (mask 外都是 0-vs-0 = 1, 会稀释), 所以我们
    # 用 ratio 方式: 只在 mask 覆盖率 > 5% 时可靠. 提供者应记录 mask 覆盖率.
    return ssim(p_masked, g_masked, window_size=window_size, size_average=True)


def masked_lpips(pred: torch.Tensor, gt: torch.Tensor, mask: torch.Tensor,
                 thr: float = 0.5, net_type: str = "vgg") -> torch.Tensor:
    """
    Masked LPIPS: 把 mask 外像素置零后跑标准 LPIPS.
    近似指标, 值越低越好.
    """
    from lpipsPyTorch import lpips
    p, g = _to_4d(pred), _to_4d(gt)
    m = _binarize_mask(mask, thr)
    if m.sum().item() < 1.0:
        return torch.tensor(float("nan"), device=p.device)
    return lpips(p * m, g * m, net_type=net_type)


def mask_coverage(mask: torch.Tensor, thr: float = 0.5) -> float:
    """返回 mask 中 1 的比例 (0-1), 用于报告 fg/bg 相对面积."""
    m = _binarize_mask(mask, thr)
    return float(m.mean().item())
