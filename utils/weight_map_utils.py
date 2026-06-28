"""
weight_map_utils.py - 水下图像可信度权重图工具

功能：
  - 从预计算的权重图目录中加载与训练图像同名的权重图
  - 或者在线从原始图像计算权重图（调用物理公式：清晰度 + UDCP 传输图）
  - 支持缓存，避免重复 I/O

使用方式：
  预计算模式（推荐）：
    先用 underwater_weight_map 工具对所有训练图像批量生成权重图，
    保存到 <source_path>/weight_maps/ 目录下（与训练图像同名），
    然后训练时自动加载。

  在线计算模式（备用）：
    直接对每张图像实时计算权重图，开销较小（纯 numpy 计算）。
"""

import os
import cv2
import numpy as np
import torch


# ─────────────────────────────────────────────
# 在线计算模块（无需预计算，调用物理公式）
# ─────────────────────────────────────────────

def compute_sharpness_weight(img_bgr: np.ndarray, sigma: float = 0.5) -> np.ndarray:
    """
    清晰度权重图：用拉普拉斯方差估计局部清晰度。
    返回值归一化到 [0, 1]，越清晰越接近 1。
    """
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
    lap = cv2.Laplacian(gray, cv2.CV_32F)
    lap_sq = lap ** 2

    kernel_size = max(3, int(6 * sigma + 1) | 1)
    local_mean = cv2.GaussianBlur(lap_sq, (kernel_size, kernel_size), sigma)
    local_mean2 = cv2.GaussianBlur(lap_sq ** 0.5, (kernel_size, kernel_size), sigma) ** 2
    sharpness = np.sqrt(np.maximum(local_mean - local_mean2, 0.0))

    s_max = sharpness.max()
    if s_max > 1e-6:
        sharpness = sharpness / s_max
    return sharpness.astype(np.float32)


def compute_transmission_weight(img_bgr: np.ndarray, patch_size: int = 15) -> np.ndarray:
    """
    传输图权重：用水下暗通道先验（UDCP）估计传输率 t(x)。
    t(x) 越大 → 该区域越靠近相机 → 可信度越高。
    返回值归一化到 [0, 1]。
    """
    img_float = img_bgr.astype(np.float32) / 255.0

    # 水下暗通道：取 R、G 通道最小值的局部最小（忽略 B 通道）
    rg_min = np.minimum(img_float[:, :, 2], img_float[:, :, 1])
    kernel = np.ones((patch_size, patch_size), dtype=np.float32)
    dark_channel = cv2.erode(rg_min, kernel)

    # 全局背景光估计
    num_pixels = dark_channel.size
    num_brightest = max(1, int(num_pixels * 0.001))
    flat_dark = dark_channel.flatten()
    bright_idx = np.argpartition(flat_dark, -num_brightest)[-num_brightest:]
    A = img_float[:, :, 0].flatten()[bright_idx].mean()
    A = np.clip(A, 0.1, 1.0)

    # 传输图估计
    omega = 0.95
    transmission = 1.0 - omega * (dark_channel / A)
    transmission = np.clip(transmission, 0.05, 1.0)

    t_min = transmission.min()
    t_max = transmission.max()
    if t_max - t_min > 1e-6:
        transmission = (transmission - t_min) / (t_max - t_min)
    return transmission.astype(np.float32)


def compute_weight_map(
    img_bgr: np.ndarray,
    alpha: float = 1.0,
    beta: float = 1.0,
    patch_size: int = 15,
) -> np.ndarray:
    """
    融合权重图：W(x) = W_sharp(x)^alpha * t(x)^beta

    Args:
        img_bgr:    BGR 格式图像，uint8
        alpha:      清晰度权重的指数
        beta:       传输率权重的指数
        patch_size: UDCP 的 patch 大小

    Returns:
        weight_map: float32, shape=(H, W)，值域 [0, 1]
    """
    w_sharp = compute_sharpness_weight(img_bgr)
    w_trans = compute_transmission_weight(img_bgr, patch_size=patch_size)
    weight = (w_sharp ** alpha) * (w_trans ** beta)

    w_max = weight.max()
    if w_max > 1e-6:
        weight = weight / w_max
    return weight.astype(np.float32)


# ─────────────────────────────────────────────
# 主接口：WeightMapLoader
# ─────────────────────────────────────────────

class WeightMapLoader:
    """
    权重图加载器，支持两种模式：
      - precomputed: 从预计算目录加载（速度快）
      - online:      实时从训练图像计算（无需预处理）
    """

    def __init__(
        self,
        source_path: str,
        mode: str = "online",
        alpha: float = 1.0,
        beta: float = 1.0,
        weight_map_dir: str = "weight_maps",
    ):
        self.source_path = source_path
        self.mode = mode
        self.alpha = alpha
        self.beta = beta
        self.weight_map_dir = os.path.join(source_path, weight_map_dir)
        self._cache = {}

        if mode == "precomputed":
            if not os.path.isdir(self.weight_map_dir):
                print(
                    f"[WeightMapLoader] WARNING: precomputed dir not found: {self.weight_map_dir}\n"
                    f"  Falling back to online mode."
                )
                self.mode = "online"
            else:
                print(f"[WeightMapLoader] Precomputed mode: {self.weight_map_dir}")
        else:
            print(f"[WeightMapLoader] Online mode (alpha={alpha}, beta={beta})")

    def get(self, viewpoint_cam, device: str = "cuda") -> torch.Tensor:
        """
        返回权重图 Tensor。

        Returns:
            weight: torch.Tensor, shape=(1, H, W), float32, 值域 [0, 1]
                    或 None（找不到预计算文件时）
        """
        img_name = viewpoint_cam.image_name

        if img_name in self._cache:
            cached = self._cache[img_name]
            return cached.to(device) if cached is not None else None

        if self.mode == "precomputed":
            weight = self._load_precomputed(img_name)
        else:
            weight = self._compute_online(viewpoint_cam)

        if weight is not None:
            weight_tensor = torch.from_numpy(weight).unsqueeze(0)  # (1, H, W)
            self._cache[img_name] = weight_tensor
            return weight_tensor.to(device)
        else:
            self._cache[img_name] = None
            return None

    def _load_precomputed(self, img_name: str):
        # 尝试 .npy（精度更高）
        npy_path = os.path.join(self.weight_map_dir, img_name + ".npy")
        if os.path.exists(npy_path):
            return np.load(npy_path).astype(np.float32)

        # 尝试同名图像文件
        for ext in [".png", ".jpg", ".jpeg"]:
            img_path = os.path.join(self.weight_map_dir, img_name + ext)
            if os.path.exists(img_path):
                w = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                if w is not None:
                    return (w.astype(np.float32) / 255.0)

        print(f"[WeightMapLoader] WARNING: not found for {img_name}, using uniform weight.")
        return None

    def _compute_online(self, viewpoint_cam):
        img_path = getattr(viewpoint_cam, "image_path", None)
        if img_path and os.path.exists(img_path):
            img_bgr = cv2.imread(img_path)
        else:
            # fallback：从 original_image tensor 反推
            img_tensor = viewpoint_cam.original_image  # (3, H, W), float [0,1]
            img_np = (img_tensor.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
            img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)

        return compute_weight_map(img_bgr, alpha=self.alpha, beta=self.beta)
