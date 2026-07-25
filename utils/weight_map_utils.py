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
# LocVar + 距离变换软填充 (2026-07-25, feat-wm-locvar-dist 分支)
# 目标: 主体填充 - 靠边缘 "影响半径" 覆盖低纹理表面 (铁柱/铁皮), 不依赖闭合
# ─────────────────────────────────────────────

def compute_local_color_var(img_bgr: np.ndarray, ksize: int = 15) -> np.ndarray:
    """
    LocVar: 3 通道局部方差之和的开方, 5-95% 分位截断归一化到 [0,1].
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


def compute_locvar_dist_weight_map(
    img_bgr: np.ndarray,
    ksize: int = 15,
    edge_thr: float = 0.4,
    sigma_ratio: float = 1.0 / 15.0,
    floor: float = 0.05,
) -> np.ndarray:
    """
    LocVar 边缘 + distanceTransform 软填充 (方案 B, 主体填充路线).

    步骤:
      1. LocVar > edge_thr 得到边缘二值图
      2. distanceTransform 拿每像素到最近边缘的欧氏距离 dist
      3. weight = exp(-dist / sigma), sigma 自适应 = min(H, W) * sigma_ratio
      4. min-max 归一 + floor

    物理意义:
      - 边缘密集处 (物体轮廓/铁皮按钮) -> dist 小 -> weight 高
      - 边缘稀疏处 (纯水体) -> dist 大 -> weight 趋近 0
      - **不依赖闭合边缘**: 铁柱边缘断裂/被遮挡也能靠 "影响半径" 覆盖主体
      - **sigma 决定覆盖尺度**: 1/15 约 ~40px, 铁柱直径量级

    调参提示 (07-25 gallery 全量 278 张统计):
      - edge_thr=0.4: LocVar 边缘阈值, 上调可减少水中光斑误抬
      - sigma_ratio=1/15: 覆盖半径, 下调 (1/20) 减少溢出但可能漏填主体内部
      - floor=0.05: 水体底噪, 保持权重范围一致

    Args:
        img_bgr:     BGR uint8 (H, W, 3)
        ksize:       LocVar 窗口大小
        edge_thr:    LocVar 二值化阈值
        sigma_ratio: sigma = min(H, W) * sigma_ratio, 覆盖半径 (影响主体填充范围)
        floor:       归一后底噪, 水体权重下限

    Returns:
        float32 (H, W), 值域 [floor, 1]
    """
    lv = compute_local_color_var(img_bgr, ksize=ksize)
    H, W = lv.shape
    edge = (lv >= edge_thr).astype(np.uint8)
    if edge.sum() == 0:
        return np.full_like(lv, floor)

    # distanceTransform: 输入 0 = 边缘/前景, 非 0 = 背景
    dist = cv2.distanceTransform(1 - edge, cv2.DIST_L2, 5)

    sigma_px = max(5.0, min(H, W) * sigma_ratio)
    weight = np.exp(-dist / sigma_px)

    wmin, wmax = float(weight.min()), float(weight.max())
    if wmax - wmin > 1e-6:
        weight = (weight - wmin) / (wmax - wmin)
    weight = floor + (1.0 - floor) * weight
    return weight.astype(np.float32)


# ─────────────────────────────────────────────
# SfM KDE 权重图 (2026-07-23, feat-wm-sfm-kde 分支新增)
# ─────────────────────────────────────────────

def _gaussian_splat_kde(H, W, xs, ys, sigma_px):
    """
    2D Gaussian KDE: 把 (xs, ys) 一批点用 sigma_px 高斯核 splash 到 H×W 网格。
    实现: 直接开 (H, W) 累加图 + cv2.GaussianBlur (等价于卷积). 稳定且快.
    返回值未归一.
    """
    heat = np.zeros((H, W), dtype=np.float32)
    if len(xs) == 0:
        return heat
    xi = np.clip(np.round(xs).astype(np.int32), 0, W - 1)
    yi = np.clip(np.round(ys).astype(np.int32), 0, H - 1)
    np.add.at(heat, (yi, xi), 1.0)
    ksize = max(3, int(6 * sigma_px + 1) | 1)
    heat = cv2.GaussianBlur(heat, (ksize, ksize), sigma_px)
    return heat


def compute_sfm_kde_weight_map(
    H: int,
    W: int,
    xys: np.ndarray,
    point3D_ids: np.ndarray,
    sigma_px: float = 25.0,
    percentile_norm: tuple = (5.0, 95.0),
    floor: float = 0.05,
) -> np.ndarray:
    """
    SfM 观测密度权重图.

    输入:
      H, W:            图像分辨率 (与训练分辨率一致, 若下采样则外面负责缩放)
      xys:             (M, 2) COLMAP 该视角的 2D 观测坐标 (像素)
      point3D_ids:     (M,) 对应 3D 点 id, <0 表示未三角化, 忽略
      sigma_px:        高斯核标准差 (像素); 越大越平滑, 前景覆盖越完整
      percentile_norm: (lo, hi) 分位截断, 抗少量异常热点
      floor:           归一后底噪 floor, 保证水体不是 0 (避免完全屏蔽)

    输出:
      W_map: float32 (H, W), 值域 [floor, 1]

    直觉:
      COLMAP 三角化成功的点 = 多视角一致的物体表面点.
      水体/浮沫: 无法三角化 -> 无点 -> KDE 值低.
      机身/铁皮: 稠密特征 -> KDE 值高.
      天然 3D 一致 (同一批 3D 点投到不同视角), 与颜色无关, 换水色不敏感.
    """
    valid = point3D_ids > 0
    xs = xys[valid, 0]
    ys = xys[valid, 1]
    heat = _gaussian_splat_kde(H, W, xs, ys, sigma_px)

    lo, hi = np.percentile(heat, percentile_norm)
    if hi - lo < 1e-6:
        return np.full((H, W), floor, dtype=np.float32)
    w = np.clip((heat - lo) / (hi - lo), 0.0, 1.0)
    w = floor + (1.0 - floor) * w
    return w.astype(np.float32)


def build_sfm_kde_maps_from_colmap(
    source_path: str,
    image_hw_lookup: dict = None,
    sigma_px: float = 25.0,
    floor: float = 0.05,
) -> dict:
    """
    一次性预计算 source_path/sparse/0 里所有视角的 SfM KDE 权重图.

    Args:
      source_path:      COLMAP scene 根 (含 sparse/0/*.bin)
      image_hw_lookup:  {image_name_stem: (H, W)}  训练分辨率. None 时用 COLMAP intrinsic 原图分辨率.
      sigma_px, floor:  透传到 compute_sfm_kde_weight_map

    Returns:
      {image_name_stem: np.float32 (H, W)}
    """
    from scene.colmap_loader import (
        read_extrinsics_binary, read_intrinsics_binary,
    )

    sparse_dir = os.path.join(source_path, "sparse", "0")
    images_bin = os.path.join(sparse_dir, "images.bin")
    cameras_bin = os.path.join(sparse_dir, "cameras.bin")
    if not (os.path.exists(images_bin) and os.path.exists(cameras_bin)):
        raise FileNotFoundError(
            f"SfM KDE 需要 {images_bin} 和 {cameras_bin}, 请先跑 COLMAP."
        )

    extrs = read_extrinsics_binary(images_bin)
    intrs = read_intrinsics_binary(cameras_bin)

    out = {}
    for img_id, extr in extrs.items():
        stem = os.path.splitext(extr.name)[0]
        cam = intrs[extr.camera_id]
        H_full, W_full = int(cam.height), int(cam.width)

        if image_hw_lookup is not None and stem in image_hw_lookup:
            H_use, W_use = image_hw_lookup[stem]
            sx = W_use / float(W_full)
            sy = H_use / float(H_full)
        else:
            H_use, W_use = H_full, W_full
            sx = sy = 1.0

        xys = extr.xys.copy()
        xys[:, 0] *= sx
        xys[:, 1] *= sy
        wmap = compute_sfm_kde_weight_map(
            H_use, W_use,
            xys=xys, point3D_ids=extr.point3D_ids,
            sigma_px=sigma_px * min(sx, sy),  # 分辨率变化时 sigma 同比缩
            floor=floor,
        )
        out[stem] = wmap

    return out


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
        method: str = "legacy",
        sfm_kde_sigma_px: float = 25.0,
        sfm_kde_floor: float = 0.05,
        locvar_dist_edge_thr: float = 0.4,
        locvar_dist_sigma_ratio: float = 1.0 / 15.0,
        locvar_dist_floor: float = 0.05,
    ):
        self.source_path = source_path
        self.mode = mode
        self.alpha = alpha
        self.beta = beta
        self.method = method
        self.sfm_kde_sigma_px = sfm_kde_sigma_px
        self.sfm_kde_floor = sfm_kde_floor
        self.locvar_dist_edge_thr = locvar_dist_edge_thr
        self.locvar_dist_sigma_ratio = locvar_dist_sigma_ratio
        self.locvar_dist_floor = locvar_dist_floor
        self.weight_map_dir = os.path.join(source_path, weight_map_dir)
        self._cache = {}
        self._sfm_kde_maps = None  # dict[stem] = (H, W) float32, 首次 get() 触发

        if method == "sfm_kde":
            print(f"[WeightMapLoader] method=sfm_kde (sigma_px={sfm_kde_sigma_px}, floor={sfm_kde_floor})")
            print(f"[WeightMapLoader]   首次 get() 时按训练分辨率一次性预计算全部帧")
            return

        if method == "locvar_dist":
            print(
                f"[WeightMapLoader] method=locvar_dist "
                f"(edge_thr={locvar_dist_edge_thr}, sigma_ratio={locvar_dist_sigma_ratio}, "
                f"floor={locvar_dist_floor}) - 主体填充路线, 距离变换软扩散, 免 SfM/K-means"
            )
            return

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
            print(f"[WeightMapLoader] method=legacy Online mode (alpha={alpha}, beta={beta})")

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

        if self.method == "sfm_kde":
            weight = self._get_sfm_kde(viewpoint_cam)
        elif self.method == "locvar_dist":
            weight = self._compute_locvar_dist(viewpoint_cam)
        elif self.mode == "precomputed":
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

    def _get_sfm_kde(self, viewpoint_cam):
        """
        延迟构建全部 SfM KDE 图: 首次调用时用 cam 的 (H, W) 决定分辨率.
        """
        img_name = viewpoint_cam.image_name  # e.g. "0001"
        img_tensor = viewpoint_cam.original_image  # (3, H, W)
        H_cur, W_cur = int(img_tensor.shape[1]), int(img_tensor.shape[2])

        if self._sfm_kde_maps is None:
            # 直接按当前分辨率一次性构建所有帧; 假设全部训练图分辨率一致
            image_hw_lookup = None
            try:
                image_hw_lookup = {img_name: (H_cur, W_cur)}
                # 用当前帧的 hw 让 builder 推 sx/sy; 其它帧假设同分辨率
                all_maps = build_sfm_kde_maps_from_colmap(
                    self.source_path,
                    image_hw_lookup={img_name: (H_cur, W_cur)},
                    sigma_px=self.sfm_kde_sigma_px,
                    floor=self.sfm_kde_floor,
                )
                # builder 里 image_hw_lookup 只对匹配到的帧生效,
                # 其它帧走 COLMAP intrinsic 原始分辨率, 这里统一按 H_cur/W_cur resize
                fixed = {}
                for stem, wm in all_maps.items():
                    if wm.shape[0] != H_cur or wm.shape[1] != W_cur:
                        wm = cv2.resize(wm, (W_cur, H_cur), interpolation=cv2.INTER_LINEAR)
                    fixed[stem] = wm
                self._sfm_kde_maps = fixed
                print(f"[WeightMapLoader/sfm_kde] built {len(fixed)} maps at {H_cur}x{W_cur}")
            except Exception as e:
                print(f"[WeightMapLoader/sfm_kde] ERROR building maps: {e}")
                self._sfm_kde_maps = {}

        wm = self._sfm_kde_maps.get(img_name, None)
        if wm is None:
            print(f"[WeightMapLoader/sfm_kde] WARN no map for {img_name}, using uniform floor")
            wm = np.full((H_cur, W_cur), self.sfm_kde_floor, dtype=np.float32)
        return wm

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
        img_bgr = self._read_bgr(viewpoint_cam)
        return compute_weight_map(img_bgr, alpha=self.alpha, beta=self.beta)

    def _compute_locvar_dist(self, viewpoint_cam):
        img_bgr = self._read_bgr(viewpoint_cam)
        return compute_locvar_dist_weight_map(
            img_bgr,
            edge_thr=self.locvar_dist_edge_thr,
            sigma_ratio=self.locvar_dist_sigma_ratio,
            floor=self.locvar_dist_floor,
        )

    def _read_bgr(self, viewpoint_cam):
        """从 cam 拿到 BGR uint8 图 (优先原图路径, fallback 到 tensor 反推)"""
        img_path = getattr(viewpoint_cam, "image_path", None)
        if img_path and os.path.exists(img_path):
            img_bgr = cv2.imread(img_path)
            if img_bgr is not None:
                return img_bgr
        # fallback: 从 original_image tensor 反推
        img_tensor = viewpoint_cam.original_image  # (3, H, W), float [0,1]
        img_np = (img_tensor.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
        return cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
