"""
基于 GaussianLOD (P6) 场景净化方案的训练时轻量集成版
======================================================

参考：/Users/yancanfeng/codeFile/GaussianLOD/src/cleaning/graph_cluster_filter.py

差异（针对训练时集成的裁剪）：
- 只保留 Stage 1 (属性预过滤) 和 Stage 3 (连通分量 + 主簇间隙)
- Stage 2 (KNN 孤立点) 完全不用（作者也默认关闭）
- 不接 GaussianLOD 的 GaussianCloud 数据类，直接对 torch 张量操作
- Stage 3 内部为了避免拖慢训练：
    * 点数 > max_points_for_graph 时子采样，采样外默认保留
    * 用 scipy.sparse.csgraph 求连通分量（CPU），避免 GPU BFS
- 所有阈值都按「相对 scene_extent」，无需项目手动调
- 返回 keep_mask（bool tensor, N），由上层决定"剪掉"或"仅统计"

设计目标：
1. 训练时低频（例如 iter ≥ 20000 时每 5000 iter 触发一次），把最终收敛的场景
   离散孤岛剪掉，让最终点云更干净
2. 保守：max_main_clusters=1，min_cluster_size=64，gap_to_main_ratio=0.05
3. 不接管 3DGS 原生的 opacity/screen-size/巨型片剪枝，只补充"空间连通性"这条判据
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np
import torch


@dataclass
class CleanConfig:
    # 总开关
    enabled: bool = False

    # Stage 1: 属性预过滤（作为 3DGS 原生 opacity/大点剪枝的补充判据）
    enable_attr: bool = True
    min_opacity: float = 0.02          # <0.02 视为"幽灵"
    max_scale_ratio: float = 0.05      # >5% scene_extent 视为巨型片

    # Stage 3: 连通分量 + 主簇间隙
    enable_cluster: bool = True
    # 触发频率控制（由外层 train 决定，模块内不管）
    eps_knn_multiplier: float = 4.0    # ε = median(1-NN) × 该倍数
    min_cluster_size: int = 64         # 簇点数 < 该值 → 判为漂浮物候选
    main_cluster_size_ratio: float = 0.1  # ≥ 10% 总点数才有资格成为主簇
    max_main_clusters: int = 1         # 只取最大 1 个作为主体，防大漂浮物晋升
    gap_to_main_ratio: float = 0.05    # 簇到主簇最近距离 > 5% extent → 判为脱节

    # 性能保护
    max_points_for_graph: int = 200_000  # 超过则子采样，未采样点默认保留
    random_seed: int = 42


def compute_keep_mask(
    xyz: torch.Tensor,           # (N, 3)
    opacity: torch.Tensor,       # (N,) 已 sigmoid（激活后）
    scale: torch.Tensor,         # (N, 3) 已 exp（激活后）
    extent: float,
    cfg: CleanConfig,
    stage: str = "both",         # "attr" | "cluster" | "both"
) -> Tuple[torch.Tensor, dict]:
    """
    Returns:
        keep_mask (N,) bool tensor on same device — True 表示应保留
        report dict — 用于日志输出
    """
    device = xyz.device
    N = xyz.shape[0]
    keep = torch.ones(N, dtype=torch.bool, device=device)
    report = {
        "input": N,
        "removed_attr": 0,
        "removed_cluster": 0,
        "cluster_count": 0,
        "main_count": 0,
        "eps_used": None,
        "subsampled": False,
    }

    if not cfg.enabled or N == 0 or extent <= 0:
        return keep, report

    # ---------- Stage 1: 属性预过滤 ----------
    if cfg.enable_attr and stage in ("attr", "both"):
        max_scale_abs = float(cfg.max_scale_ratio) * float(extent)
        smax = scale.max(dim=1).values
        attr_bad = (opacity < float(cfg.min_opacity)) | (smax > max_scale_abs)
        keep &= ~attr_bad
        report["removed_attr"] = int(attr_bad.sum().item())

    # 提前终止
    if int(keep.sum().item()) == 0:
        return keep, report

    # ---------- Stage 3: 连通分量 + 主簇间隙 ----------
    if cfg.enable_cluster and stage in ("cluster", "both"):
        alive_idx = torch.nonzero(keep, as_tuple=False).squeeze(-1)  # (M,)
        M = alive_idx.numel()
        if M < max(2, cfg.min_cluster_size):
            return keep, report

        # 子采样（CPU 拷贝）
        if M > cfg.max_points_for_graph:
            rng = np.random.default_rng(cfg.random_seed)
            sub_local = rng.choice(M, size=cfg.max_points_for_graph, replace=False)
            sub_local = np.sort(sub_local)  # 保序方便 searchsorted
            report["subsampled"] = True
        else:
            sub_local = np.arange(M)

        pos_cpu = xyz[alive_idx[sub_local]].detach().cpu().numpy().astype(np.float32)
        Msub = pos_cpu.shape[0]

        # KNN: 只需要"1-NN 距离"用于自适应 ε
        d1 = _knn_1st_distance(pos_cpu)
        if d1.size == 0:
            return keep, report
        median_1nn = float(np.median(d1))
        eps_abs = max(median_1nn * float(cfg.eps_knn_multiplier), 1e-9)
        report["eps_used"] = eps_abs

        # 连通分量
        labels, n_clusters = _connected_components(pos_cpu, eps_abs)
        report["cluster_count"] = n_clusters
        if n_clusters == 0:
            return keep, report

        cluster_sizes = np.bincount(labels, minlength=n_clusters)
        total = int(cluster_sizes.sum())
        main_threshold = max(1, int(np.ceil(cfg.main_cluster_size_ratio * total)))
        eligible = cluster_sizes >= main_threshold

        # 主簇 = eligible ∩ top-K
        if cfg.max_main_clusters > 0 and eligible.sum() > cfg.max_main_clusters:
            top_idx = np.argsort(-cluster_sizes)[: cfg.max_main_clusters]
            is_main = np.zeros(n_clusters, dtype=bool)
            is_main[top_idx] = True
            is_main &= eligible
        else:
            is_main = eligible
        report["main_count"] = int(is_main.sum())

        candidate = ~is_main
        if not candidate.any():
            return keep, report

        # 判据 A：小簇
        bad_small = candidate & (cluster_sizes < cfg.min_cluster_size)

        # 判据 B：离主簇最近距离 > 阈值
        bad_gap = np.zeros(n_clusters, dtype=bool)
        if report["main_count"] > 0:
            from scipy.spatial import cKDTree
            main_pts = pos_cpu[np.isin(labels, np.nonzero(is_main)[0])]
            if main_pts.shape[0] > 0:
                tree = cKDTree(main_pts)
                gap_thr = float(cfg.gap_to_main_ratio) * float(extent)
                for cid in np.nonzero(candidate)[0]:
                    pts = pos_cpu[labels == cid]
                    if pts.shape[0] == 0:
                        continue
                    d, _ = tree.query(pts, k=1, workers=-1)
                    if float(d.min()) > gap_thr:
                        bad_gap[cid] = True

        bad_cluster = bad_small | bad_gap
        bad_labels = np.nonzero(bad_cluster)[0]
        if bad_labels.size > 0:
            bad_point_local = np.isin(labels, bad_labels)  # (Msub,) bool
            if bad_point_local.any():
                # sub_local -> alive_idx 位置 -> 原 xyz 索引
                bad_alive_local = sub_local[bad_point_local]  # 相对 alive 的下标
                bad_orig = alive_idx[torch.from_numpy(bad_alive_local).to(device)]
                keep[bad_orig] = False
                report["removed_cluster"] = int(bad_orig.numel())

    report["output"] = int(keep.sum().item())
    return keep, report


def _knn_1st_distance(points: np.ndarray) -> np.ndarray:
    """返回每个点到最近邻（非自身）的距离 (N,)"""
    n = points.shape[0]
    if n <= 1:
        return np.zeros(n, dtype=np.float32)
    try:
        from sklearn.neighbors import NearestNeighbors
        nn = NearestNeighbors(n_neighbors=2, algorithm="auto", n_jobs=-1)
        nn.fit(points)
        d, _ = nn.kneighbors(points, return_distance=True)
        return d[:, 1].astype(np.float32, copy=False)
    except ImportError:
        pass
    from scipy.spatial import cKDTree
    tree = cKDTree(points)
    d, _ = tree.query(points, k=2, workers=-1)
    if d.ndim == 1:
        d = d[:, None]
    return d[:, 1].astype(np.float32, copy=False)


def _connected_components(points: np.ndarray, eps: float):
    """ε-邻域图 + 连通分量。返回 (labels, n_clusters)。"""
    from scipy.sparse import csr_matrix
    from scipy.sparse.csgraph import connected_components
    n = points.shape[0]
    if n == 0:
        return np.zeros(0, dtype=np.int32), 0
    if n == 1:
        return np.zeros(1, dtype=np.int32), 1
    try:
        from sklearn.neighbors import radius_neighbors_graph
        adj = radius_neighbors_graph(
            points, radius=eps, mode="connectivity",
            include_self=False, n_jobs=-1,
        )
    except ImportError:
        from scipy.spatial import cKDTree
        tree = cKDTree(points)
        pairs = tree.query_pairs(r=eps, output_type="ndarray")
        if pairs.size == 0:
            return np.arange(n, dtype=np.int32), n
        rows = np.concatenate([pairs[:, 0], pairs[:, 1]])
        cols = np.concatenate([pairs[:, 1], pairs[:, 0]])
        data = np.ones(rows.size, dtype=np.uint8)
        adj = csr_matrix((data, (rows, cols)), shape=(n, n))
    n_comp, labels = connected_components(adj, directed=False, return_labels=True)
    return labels.astype(np.int32, copy=False), int(n_comp)
