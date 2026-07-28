"""
medium_model.py - 水下介质物理模型 (M4, IntentSplat C1)

渲染方程:
  I_fg = J * t + B_inf * (1 - t)
  t = exp(-beta * depth)

可学参数:
  beta:  3 通道衰减系数 (R > G > B for underwater)
  B_inf: 3 通道散射光 (背景光/水色)

约束:
  beta = softplus(beta_raw) + eps  (正性)
  B_inf = sigmoid(B_raw)           (值域 [0, 1])
  L_mono = ReLU(beta_G - beta_R) + ReLU(beta_B - beta_G)  (通道单调)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class MediumModel(nn.Module):

    def __init__(self, beta_init=(1.5, 0.5, 0.3), B_inf_init=(0.1, 0.3, 0.5)):
        super().__init__()
        beta_t = torch.tensor(beta_init, dtype=torch.float32)
        self.beta_raw = nn.Parameter(self._inv_softplus(beta_t))

        B_t = torch.tensor(B_inf_init, dtype=torch.float32)
        self.B_raw = nn.Parameter(torch.log(B_t / (1.0 - B_t + 1e-6)))

    @staticmethod
    def _inv_softplus(x, threshold=20.0):
        return torch.where(x > threshold, x, x.expm1().log())

    @property
    def beta(self):
        return F.softplus(self.beta_raw) + 1e-2

    @property
    def B_inf(self):
        return torch.sigmoid(self.B_raw)

    def forward(self, J, depth):
        """
        Args:
            J:     (3, H, W) 高斯渲染的 radiance (去介质后的物体真色)
            depth: (1, H, W) 渲染深度

        Returns:
            I_fg:  (3, H, W) 前景成像 = J * t + B * (1-t)
        """
        beta = self.beta.view(3, 1, 1)       # (3, 1, 1)
        B = self.B_inf.view(3, 1, 1)          # (3, 1, 1)
        t = torch.exp(-beta * depth)          # (3, H, W)
        return J * t + B * (1.0 - t)

    def mono_loss(self):
        """通道单调约束: beta_R > beta_G > beta_B (水下物理先验)"""
        b = self.beta
        return F.relu(b[1] - b[0]) + F.relu(b[2] - b[1])

    def init_B_from_bg_pixels(self, gt_image, mask):
        """
        从 GT 图像的 bg 像素估计 B_inf 初始值.

        Args:
            gt_image: (3, H, W) GT
            mask:     (1, H, W) 前景 mask (1=fg, 0=bg)
        """
        with torch.no_grad():
            bg_mask = (mask < 0.5)  # (1, H, W)
            medians = []
            for c in range(3):
                ch = gt_image[c][bg_mask[0]]
                if ch.numel() > 0:
                    medians.append(ch.median())
                else:
                    medians.append(torch.tensor(0.3, device=gt_image.device))
            B_target = torch.stack(medians).to(gt_image.device)
            B_target = B_target.clamp(0.01, 0.99)
            self.B_raw.data.copy_(torch.log(B_target / (1.0 - B_target)))

    def init_B_from_gt(self, gt_image):
        """
        无 mask 时从 GT 图像整体估计 B_inf (取每通道中位数).
        适用于 SeathruNeRF 等全水下场景.
        """
        with torch.no_grad():
            medians = [gt_image[c].median() for c in range(3)]
            B_target = torch.stack(medians).to(gt_image.device).clamp(0.01, 0.99)
            self.B_raw.data.copy_(torch.log(B_target / (1.0 - B_target)))
