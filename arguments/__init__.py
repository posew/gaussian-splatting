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
        self.weight_prune_thr = 0.2  # (deprecated) 旧权重图剪枝比例, 已弃用
        self.use_normal_loss = False
        self.lambda_normal = 0.01
        # ── 剪枝参数：v3 各向异性剪枝（v4 默认关闭）──
        self.aniso_thr = 0.0               # v4: 关闭各向异性剪枝（>1.0 时才生效）
        self.aniso_min_scale_ratio = 0.001 # 各向异性剪枝的最小尺度保护(相对 scene_extent)
        self.contrib_prune_thr = 0.0       # v3: 关闭贡献度剪枝
        # ── v4: 场景净化剪枝（GaussianLOD P6 方案的训练时轻量集成版）──
        self.clean_enabled = False         # v4b 之后默认关闭；场景净化路径已证明不 work
        self.clean_attr_min_opacity = 0.02 # Stage 1: opacity 阈值
        self.clean_attr_max_scale_ratio = 0.05  # Stage 1: 巨型片阈值 (× extent)
        # v4b: Stage1 节流（避免每次 densify 都触发导致过剪）
        self.clean_attr_from_iter = 3000        # Stage 1 起始 iter
        self.clean_attr_every = 10              # Stage 1 每 N 次 densify 才触发一次
        self.clean_cluster_from_iter = 20000    # Stage 3 起始 iter (末期才跑)
        self.clean_cluster_interval = 5000      # Stage 3 触发间隔
        self.clean_cluster_min_size = 64        # Stage 3: 小簇阈值
        self.clean_cluster_gap_ratio = 0.05     # Stage 3: 到主簇最近距离阈值 (× extent)
        self.clean_cluster_eps_multiplier = 4.0 # Stage 3: 自适应 ε = median(1-NN) × 该倍数
        self.clean_cluster_max_points = 200000  # Stage 3: 建图前的子采样上限
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
