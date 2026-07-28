#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

from argparse import ArgumentParser, Namespace
import sys
import os

class GroupParams:
    pass

class ParamGroup:
    def __init__(self, parser: ArgumentParser, name : str, fill_none = False):
        group = parser.add_argument_group(name)
        for key, value in vars(self).items():
            shorthand = False
            if key.startswith("_"):
                shorthand = True
                key = key[1:]
            t = type(value)
            value = value if not fill_none else None 
            if shorthand:
                if t == bool:
                    group.add_argument("--" + key, ("-" + key[0:1]), default=value, action="store_true")
                else:
                    group.add_argument("--" + key, ("-" + key[0:1]), default=value, type=t)
            else:
                if t == bool:
                    group.add_argument("--" + key, default=value, action="store_true")
                else:
                    group.add_argument("--" + key, default=value, type=t)

    def extract(self, args):
        group = GroupParams()
        for arg in vars(args).items():
            if arg[0] in vars(self) or ("_" + arg[0]) in vars(self):
                setattr(group, arg[0], arg[1])
        return group

class ModelParams(ParamGroup): 
    def __init__(self, parser, sentinel=False):
        self.sh_degree = 3
        self._source_path = ""
        self._model_path = ""
        self._images = "images"
        self._depths = ""
        self._resolution = -1
        self._white_background = False
        self.train_test_exp = False
        self.data_device = "cuda"
        self.eval = False
        super().__init__(parser, "Loading Parameters", sentinel)

    def extract(self, args):
        g = super().extract(args)
        g.source_path = os.path.abspath(g.source_path)
        return g

class PipelineParams(ParamGroup):
    def __init__(self, parser):
        self.convert_SHs_python = False
        self.compute_cov3D_python = False
        self.debug = False
        self.antialiasing = False
        super().__init__(parser, "Pipeline Parameters")

class OptimizationParams(ParamGroup):
    def __init__(self, parser):
        self.iterations = 30_000
        self.position_lr_init = 0.00016
        self.position_lr_final = 0.0000016
        self.position_lr_delay_mult = 0.01
        self.position_lr_max_steps = 30_000
        self.feature_lr = 0.0025
        self.opacity_lr = 0.025
        self.scaling_lr = 0.005
        self.rotation_lr = 0.001
        self.exposure_lr_init = 0.01
        self.exposure_lr_final = 0.001
        self.exposure_lr_delay_steps = 0
        self.exposure_lr_delay_mult = 0.0
        self.percent_dense = 0.01
        self.lambda_dssim = 0.2
        self.densification_interval = 100
        self.opacity_reset_interval = 3000
        self.densify_from_iter = 500
        self.densify_until_iter = 15_000
        self.densify_grad_threshold = 0.0002
        self.depth_l1_weight_init = 1.0
        self.depth_l1_weight_final = 0.01
        self.random_background = False
        self.optimizer_type = "default"
        # -- weight-map guided training --
        # W(x) = sharpness(x)^alpha * UDCP_transmission(x)^beta，归一化到 [0, 1]
        # 两个作用点独立开关:
        #   1) L1 loss 加权 (软引导, 由 use_weight_map 控制)
        #   2) Densify 硬门控 (硬抑制, 由 weight_densify_thr > 0 触发)
        # 07-19_03 · 恢复 07-05 initial fbe0dfc 的 densify 硬门控 (weight_densify_thr=0.3)
        self.use_weight_map = False            # 是否启用 (loss 加权 + 门控前置条件)
        self.weight_map_mode = "online"        # "online" | "precomputed"
        self.weight_map_alpha = 1.0            # 清晰度指数
        self.weight_map_beta = 1.0             # UDCP 传输率指数
        self.weight_densify_thr = 0.0          # densify 硬门控阈值 (0=关闭=loss-only, 0.3=initial fbe0dfc 初衷值)
        # -- SfM KDE 权重图 (feat-wm-sfm-kde, 2026-07-23) --
        # method="sfm_kde":    用 COLMAP 三角化点做 2D 高斯 KDE, 天然 3D 一致, 免标注
        # method="locvar_dist": LocVar 边缘 + distanceTransform 软填充, 主体填充路线
        # method="wm_v2":      M1 (2026-07-25): kmeans_ab × (1-caustic) ∪ colmap_seed
        self.weight_map_method = "legacy"      # "legacy" | "sfm_kde" | "locvar_dist" | "wm_v2"
        self.sfm_kde_sigma_px = 25.0           # 高斯核标准差 (像素, 训练分辨率下)
        self.sfm_kde_floor = 0.05              # 归一后底噪 floor, 避免水体权重 0
        # -- LocVar + 距离变换 (feat-wm-locvar-dist, 2026-07-25) --
        self.locvar_dist_edge_thr = 0.4        # LocVar 边缘二值化阈值
        self.locvar_dist_sigma_ratio = 1.0/15  # sigma = min(H,W) * ratio, 覆盖半径
        self.locvar_dist_floor = 0.05          # 归一后底噪 floor
        # -- wm_v2 (M1, 2026-07-25) --
        self.wm_v2_kmeans_k = 16               # LAB (a,b) K-means 簇数
        self.wm_v2_caustic_L = 200             # 光斑亮度阈值 (LAB L 通道), 水下场景 L_p99≈230
        self.wm_v2_caustic_chroma = 40         # 光斑色度阈值 (|a-128|+|b-128|), 整体色度低故放宽
        self.wm_v2_caustic_dilate = 9          # 光斑 mask 扩边像素
        self.wm_v2_colmap_seed_radius = 15     # COLMAP 每个种子画圆的半径
        self.wm_v2_use_colmap_seed = True      # 是否融合 COLMAP 种子
        # -- M3: 三层门控 + 显式背景色 (feat-tri-gate-bg, 2026-07-26) --
        self.wm_hard_thr = 0.3                 # hard mask 二值化阈值
        self.wm_erode_px = 6                   # mask 边界腐蚀像素 (SSIM 窗口一半)
        self.wm_prune_interval = 500           # prune 门控触发间隔 (iter)
        self.wm_prune_decay = 0.5              # bg 高斯 opacity 半衰系数
        self.use_explicit_bg = False            # 启用 per-view 显式背景色
        # -- M4: 水下介质模型 C2 (分离衰减/散射, 参考 SeaSplat) --
        self.use_medium = False                # 启用水下介质模型
        self.c2_from_iter = 15000              # 介质模型启动迭代
        self.c2_init_iters = 1000              # 初始化阶段长度 (冻结 GS 几何)
        self.beta_attn_init_r = 2.5            # 衰减 β 初始值 R
        self.beta_attn_init_g = 2.0            # 衰减 β 初始值 G
        self.beta_attn_init_b = 1.5            # 衰减 β 初始值 B
        self.beta_bs_init_r = 1.5              # 散射 β 初始值 R
        self.beta_bs_init_g = 1.2              # 散射 β 初始值 G
        self.beta_bs_init_b = 1.0              # 散射 β 初始值 B
        self.medium_lr = 0.01                  # 介质参数学习率
        self.w_mono = 0.1                      # 通道单调约束权重
        self.w_dcp = 1.0                       # DCP loss 权重
        self.w_gw = 0.1                        # Gray World loss 权重
        super().__init__(parser, "Optimization Parameters")

def get_combined_args(parser : ArgumentParser):
    cmdlne_string = sys.argv[1:]
    cfgfile_string = "Namespace()"
    args_cmdline = parser.parse_args(cmdlne_string)

    try:
        cfgfilepath = os.path.join(args_cmdline.model_path, "cfg_args")
        print("Looking for config file in", cfgfilepath)
        with open(cfgfilepath) as cfg_file:
            print("Config file found: {}".format(cfgfilepath))
            cfgfile_string = cfg_file.read()
    except TypeError:
        print("Config file not found at")
        pass
    args_cfgfile = eval(cfgfile_string)

    merged_dict = vars(args_cfgfile).copy()
    for k,v in vars(args_cmdline).items():
        if v != None:
            merged_dict[k] = v
    return Namespace(**merged_dict)
