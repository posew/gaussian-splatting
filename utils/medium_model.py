"""
medium_model.py - 水下介质物理模型 (IntentSplat)

C1 (MediumModel):  共享 β, 已弃用
C2 (MediumModelV2): 分离衰减/散射, 参考 SeaSplat

C2 渲染方程:
  I_uw = J * exp(-β_attn * d) + B∞ * (1 - exp(-β_bs * d))

可学参数:
  β_attn: 3 通道衰减系数 (R > G > B)
  β_bs:   3 通道散射系数 (R > G > B)
  B∞:     3 通道背景光
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class MediumModel(nn.Module):
    """C1: 共享 β 介质模型 (保留向后兼容)"""

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
        beta = self.beta.view(3, 1, 1)
        B = self.B_inf.view(3, 1, 1)
        t = torch.exp(-beta * depth)
        return J * t + B * (1.0 - t)

    def mono_loss(self):
        b = self.beta
        return F.relu(b[1] - b[0]) + F.relu(b[2] - b[1])

    def init_B_from_bg_pixels(self, gt_image, mask):
        with torch.no_grad():
            bg_mask = (mask < 0.5)
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
        with torch.no_grad():
            medians = [gt_image[c].median() for c in range(3)]
            B_target = torch.stack(medians).to(gt_image.device).clamp(0.01, 0.99)
            self.B_raw.data.copy_(torch.log(B_target / (1.0 - B_target)))


class MediumModelV2(nn.Module):
    """C2: 分离衰减/散射介质模型 (参考 SeaSplat)"""

    def __init__(self, beta_attn_init=(2.5, 2.0, 1.5),
                 beta_bs_init=(1.5, 1.2, 1.0)):
        super().__init__()
        attn_t = torch.tensor(beta_attn_init, dtype=torch.float32)
        self.beta_attn_raw = nn.Parameter(self._inv_softplus(attn_t))

        bs_t = torch.tensor(beta_bs_init, dtype=torch.float32)
        self.beta_bs_raw = nn.Parameter(self._inv_softplus(bs_t))

        self.B_raw = nn.Parameter(torch.zeros(3))

    @staticmethod
    def _inv_softplus(x, threshold=20.0):
        return torch.where(x > threshold, x, x.expm1().log())

    @property
    def beta_attn(self):
        return F.softplus(self.beta_attn_raw).clamp(max=5.0) + 1e-2

    @property
    def beta_bs(self):
        return F.softplus(self.beta_bs_raw).clamp(max=5.0) + 1e-2

    @property
    def B_inf(self):
        return torch.sigmoid(self.B_raw)

    def forward(self, J, depth_norm):
        """
        Args:
            J:          (3, H, W) 高斯渲染的 radiance
            depth_norm: (1, H, W) 归一化深度 [0, 1]
        Returns:
            I_uw: (3, H, W) = J * attenuation + backscatter
        """
        ba = self.beta_attn.view(3, 1, 1)
        bb = self.beta_bs.view(3, 1, 1)
        B = self.B_inf.view(3, 1, 1)
        attn = torch.exp(-ba * depth_norm)
        bs = B * (1.0 - torch.exp(-bb * depth_norm))
        return J * attn + bs

    def backscatter(self, depth_norm):
        """仅散射项, 用于 DCP loss"""
        bb = self.beta_bs.view(3, 1, 1)
        B = self.B_inf.view(3, 1, 1)
        return B * (1.0 - torch.exp(-bb * depth_norm))

    def mono_loss(self):
        """通道单调约束: R > G > B (衰减和散射都约束)"""
        loss = torch.tensor(0.0, device=self.beta_attn_raw.device)
        for b in [self.beta_attn, self.beta_bs]:
            loss = loss + F.relu(b[1] - b[0]) + F.relu(b[2] - b[1])
        return loss

    def init_B_from_gt(self, gt_image):
        """从 GT 图像整体估计 B∞ (取每通道中位数)"""
        with torch.no_grad():
            medians = [gt_image[c].median() for c in range(3)]
            B_target = torch.stack(medians).to(gt_image.device).clamp(0.01, 0.99)
            self.B_raw.data.copy_(torch.log(B_target / (1.0 - B_target)))

    def init_B_from_bg_pixels(self, gt_image, mask):
        """从 GT 图像的 bg 像素估计 B∞"""
        with torch.no_grad():
            bg_mask = (mask < 0.5)
            medians = []
            for c in range(3):
                ch = gt_image[c][bg_mask[0]]
                if ch.numel() > 0:
                    medians.append(ch.median())
                else:
                    medians.append(torch.tensor(0.3, device=gt_image.device))
            B_target = torch.stack(medians).to(gt_image.device).clamp(0.01, 0.99)
            self.B_raw.data.copy_(torch.log(B_target / (1.0 - B_target)))
