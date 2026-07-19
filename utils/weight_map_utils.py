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


def compute_local_color_var(img_bgr: np.ndarray, ksize: int = 15) -> np.ndarray:
    """
    Local Color Variance: 3 通道局部方差之和的开方, 5-95% 分位截断归一化。
    高: 颜色/纹理跳变强 (物体边缘、细节)
    低: 颜色平坦 (水体、铁皮平面)

    (2026-07-19 从 mini-splatting fix-weightmap-norm @ 11415de 移植)
    """
    f = img_bgr.astype(np.float32) / 255.0
    var_total = np.zeros(f.shape[:2], np.float32)
    for c in range(3):
        ch = f[:, :, c]
        mean = cv2.blur(ch, (ksize, ksize))
        mean_sq = cv2.blur(ch ** 2, (ksize, ksize))
        var_total += np.maximum(mean_sq - mean ** 2, 0.0)
    v = np.sqrt(var_total)
    lo, hi = np.percentile(v, [5, 95])
    return np.clip((v - lo) / max(hi - lo, 1e-6), 0.0, 1.0).astype(np.float32)


def compute_kmeans_weight_map(
    img_bgr: np.ndarray,
    ksize: int = 15,
    k: int = 16,
    kmeans_attempts: int = 3,
    kmeans_max_iter: int = 10,
) -> np.ndarray:
    """
    K-means 颜色聚类 + 类内 LocVar mean (mini-splatting v7 定稿, 2026-07-11)。

    核心思想 (用户洞察):
      - 铁皮的按钮 (LocVar 高) 和铁皮的平面 (LocVar 低) 颜色一致
      - 水体和物体颜色差别大
      → 按颜色分类, 每类 LocVar mean 就能同时标亮 "按钮 + 铁皮平面",
         并让水体保持暗
      → 相当于用颜色相似性做 "水下区域联通识别" (无空间约束的区域聚类)

    步骤:
      1. 计算 LocVar (compute_local_color_var)
      2. 在 Lab 空间对像素做 K-means (K=16, 类别多则颜色区分更细)
      3. 每个类内取 LocVar mean 作为该类分数
      4. 类分数 min-max 归一化到 [0, 1]
      5. 把类分数广播回像素得到 weight map

    为什么不用 "类内 max": 每类都有可能包含少量高纹理 outlier, 用 max
      会导致所有类都变 1.0 (mini v7 首次尝试实测塌陷)。mean 更稳。

    Args:
        img_bgr:         BGR 格式图像, uint8
        ksize:           LocVar 局部窗口大小
        k:               K-means 类别数 (16 更保守; 8 更激进)
        kmeans_attempts: K-means 重启次数 (取最优)
        kmeans_max_iter: K-means 最大迭代次数

    Returns:
        weight_map: float32, shape=(H, W), 值域 [0, 1]

    (2026-07-19 从 mini-splatting fix-weightmap-norm @ 11415de 移植)
    """
    lv = compute_local_color_var(img_bgr, ksize)
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    H, W = lab.shape[:2]
    pixels = lab.reshape(-1, 3)

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
                kmeans_max_iter, 1.0)
    _, labels, _ = cv2.kmeans(
        pixels, k, None, criteria, kmeans_attempts, cv2.KMEANS_PP_CENTERS
    )
    labels = labels.flatten()
    flat_lv = lv.flatten()

    # 每类 mean = sum / cnt
    class_sum = np.zeros(k, np.float64)
    class_cnt = np.zeros(k, np.float64)
    np.add.at(class_sum, labels, flat_lv)
    np.add.at(class_cnt, labels, 1)
    class_mean = (class_sum / np.maximum(class_cnt, 1)).astype(np.float32)

    # 类均值 min-max 归一化到 [0,1]
    cmin, cmax = class_mean.min(), class_mean.max()
    if cmax - cmin > 1e-6:
        class_mean = (class_mean - cmin) / (cmax - cmin)

    weight = class_mean[labels].reshape(H, W)
    return np.clip(weight, 0.0, 1.0).astype(np.float32)


def compute_weight_map(
    img_bgr: np.ndarray,
    alpha: float = 1.0,
    beta: float = 1.0,
    patch_size: int = 15,
    method: str = "kmeans",
    kmeans_k: int = 16,
    kmeans_ksize: int = 15,
) -> np.ndarray:
    """
    权重图统一入口。

    method="kmeans" (推荐, 从 mini-splatting v7 移植, 2026-07-19):
      调用 compute_kmeans_weight_map: K-means 颜色聚类 + 类内 LocVar mean
      - 铁皮按钮 (高纹理) → 高分类 → 类内平面被一起标亮
      - 水体 (低纹理 & 独立颜色) → 低分类 → 保持暗
      - LOW PSNR 图: med≈0.3, HIGH PSNR 图: med≈0.1 (符合语义)

    method="legacy" (老公式, 仅供对比, 有两个 bug):
      W(x) = W_sharp(x)^alpha * t(x)^beta
      - bug1: sharpness /= max → 全图 med≈0.05
      - bug2: transmission min-max → 全图 med≈1.0 (等于没起作用)
      → 组合结果 med≈0.05

    Args:
        img_bgr:      BGR 格式, uint8
        method:       "kmeans" | "legacy"
        kmeans_k:     K-means 类数 (仅 method=kmeans)
        kmeans_ksize: LocVar 窗口 (仅 method=kmeans)
        alpha/beta/patch_size: 仅 method=legacy 有效

    Returns:
        weight_map: float32, shape=(H, W), 值域 [0, 1]
    """
    if method == "kmeans":
        return compute_kmeans_weight_map(
            img_bgr, ksize=kmeans_ksize, k=kmeans_k
        )
    elif method == "legacy":
        w_sharp = compute_sharpness_weight(img_bgr)
        w_trans = compute_transmission_weight(img_bgr, patch_size=patch_size)
        weight = (w_sharp ** alpha) * (w_trans ** beta)
        w_max = weight.max()
        if w_max > 1e-6:
            weight = weight / w_max
        return weight.astype(np.float32)
    else:
        raise ValueError(f"unknown method: {method!r}, expected 'kmeans' or 'legacy'")


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
        method: str = "kmeans",
        kmeans_k: int = 16,
    ):
        self.source_path = source_path
        self.mode = mode
        self.alpha = alpha
        self.beta = beta
        self.method = method
        self.kmeans_k = kmeans_k
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
            print(
                f"[WeightMapLoader] Online mode "
                f"(method={method}, kmeans_k={kmeans_k}, alpha={alpha}, beta={beta})"
            )

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

        return compute_weight_map(
            img_bgr,
            alpha=self.alpha,
            beta=self.beta,
            method=self.method,
            kmeans_k=self.kmeans_k,
        )
