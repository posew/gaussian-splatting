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

import os
import torch
from random import randint
from utils.loss_utils import l1_loss, ssim
from gaussian_renderer import render, network_gui
import sys
from scene import Scene, GaussianModel
from utils.general_utils import safe_state, get_expon_lr_func
from utils.weight_map_utils import WeightMapLoader
import uuid
from tqdm import tqdm
from utils.image_utils import psnr
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams
try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False

try:
    from fused_ssim import fused_ssim
    FUSED_SSIM_AVAILABLE = True
except:
    FUSED_SSIM_AVAILABLE = False

try:
    from diff_gaussian_rasterization import SparseGaussianAdam
    SPARSE_ADAM_AVAILABLE = True
except:
    SPARSE_ADAM_AVAILABLE = False

def training(dataset, opt, pipe, testing_iterations, saving_iterations, checkpoint_iterations, checkpoint, debug_from):

    if not SPARSE_ADAM_AVAILABLE and opt.optimizer_type == "sparse_adam":
        sys.exit(f"Trying to use sparse adam but it is not installed, please install the correct rasterizer using pip install [3dgs_accel].")

    first_iter = 0
    tb_writer = prepare_output_and_logger(dataset)
    gaussians = GaussianModel(dataset.sh_degree, opt.optimizer_type)
    scene = Scene(dataset, gaussians)
    gaussians.training_setup(opt)
    if checkpoint:
        (model_params, first_iter) = torch.load(checkpoint)
        gaussians.restore(model_params, opt)

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    iter_start = torch.cuda.Event(enable_timing = True)
    iter_end = torch.cuda.Event(enable_timing = True)

    use_sparse_adam = opt.optimizer_type == "sparse_adam" and SPARSE_ADAM_AVAILABLE
    depth_l1_weight = get_expon_lr_func(opt.depth_l1_weight_init, opt.depth_l1_weight_final, max_steps=opt.iterations)

    # ------ weight-map guided training ------
    # 两个作用点 (可独立开关):
    #   1) L1 loss 加权 (软引导): Ll1 = (W * |img - gt|).mean()
    #   2) Densify 硬门控 (硬抑制): 只有 W(投影像素) > weight_densify_thr 的高斯
    #      才计入 add_densification_stats -> 从源头阻止低权重区 (水体) densify 繁殖
    # 07-19_03 · 恢复 07-05 initial fbe0dfc 的初衷机制 (densify 硬门控)
    # weight_densify_thr=0.0 (默认) 关闭门控, 等价于 loss-only 行为
    weight_map_loader = None
    if opt.use_weight_map:
        weight_map_loader = WeightMapLoader(
            source_path=dataset.source_path,
            mode=opt.weight_map_mode,
            alpha=opt.weight_map_alpha,
            beta=opt.weight_map_beta,
        )
        print(
            f"[weightmap] Enabled: mode={opt.weight_map_mode}, "
            f"alpha={opt.weight_map_alpha}, beta={opt.weight_map_beta}, "
            f"densify_thr={opt.weight_densify_thr}"
            + (" (loss-only)" if opt.weight_densify_thr <= 0 else " (loss + hard densify gate)")
        )

    if getattr(opt, "lambda_aniso", 0.0) > 0:
        print(f"[aniso-reg] Enabled: mode={getattr(opt,'aniso_mode','hard')}, "
              f"lambda_aniso={opt.lambda_aniso}, max_ratio={opt.aniso_max_ratio}")

    if getattr(opt, "wm_hard_prune_thr", 0.0) > 0:
        _grid = getattr(opt, "wm_hard_prune_grid", 5)
        _rmul = getattr(opt, "wm_hard_prune_radius_mul", 2.0)
        _oflw = getattr(opt, "wm_hard_prune_overflow_ratio", 0.7)
        print(f"[hard-prune] Enabled: thr={opt.wm_hard_prune_thr}, "
              f"interval={opt.wm_hard_prune_interval}, "
              f"grid={_grid}x{_grid}, radius_mul={_rmul}, overflow_ratio={_oflw}, "
              f"多点椭圆采样判定, 所有视角外溢占比 > {_oflw} 才剔除")

    # -------- 07-20_02 · 物理删除非重点区高斯 (hard prune) 辅助函数 --------
    # 2026-07-22 增强: 从单点判定改为投影椭圆多点采样 (5x5=25 点)
    #   动机: 中心点在重点区、但椭球投影覆盖越出重点区的高斯 (骑墙/雾伞/横穿型)
    #         用单点判定时会漏网, 导致水体上仍有大量雾状高斯
    #   方案: 用 render 时的 radii (屏幕空间半径, 像素) 在椭圆包围盒内采样 25 点,
    #         每个视角统计"投影范围内 wm<thr 的像素比例", 累积各视角最小的"外溢占比";
    #         若最小外溢占比仍 > overflow_ratio => 说明所有视角都严重外溢 => 剔除
    def wm_hard_prune_step(gaussians, scene, weight_map_loader, thr, iteration,
                            grid_size=5, radius_mul=2.0, overflow_ratio=0.7,
                            pipe=None, background=None):
        """
        多点采样判定 (方案 A):
          对每个高斯, 每个可见视角:
            1) 用 render 拿到该视角下每个高斯的屏幕空间半径 (radii, 像素单位)
            2) 在中心 ± radius_mul * radii 范围内做 grid_size x grid_size 均匀采样
            3) 统计 25 个采样点中 wm < thr 的比例 = 该视角的"外溢占比"
          聚合所有可见视角, 取"外溢占比最小的那个视角" (即最能证明这个高斯在重点区):
            min_overflow_ratio[i] > overflow_ratio => 所有视角都严重外溢 => 物理删除
        """
        with torch.no_grad():
            N = gaussians.get_xyz.shape[0]
            device = gaussians.get_xyz.device
            # 初始化为 1.0 (从未可见的默认最大外溢); 后面 visible_in_any 会保护它们不被误删
            min_overflow = torch.ones(N, device=device)
            visible_in_any = torch.zeros(N, dtype=torch.bool, device=device)

            train_cams = scene.getTrainCameras()
            xyz = gaussians.get_xyz
            ones_col = torch.ones(N, 1, device=device)
            xyz_h = torch.cat([xyz, ones_col], dim=1)  # (N, 4)

            # grid_size x grid_size 采样偏移 (单位: radii 倍数, 范围 [-radius_mul, +radius_mul])
            offs = torch.linspace(-radius_mul, radius_mul, grid_size, device=device)
            oy, ox = torch.meshgrid(offs, offs, indexing="ij")
            ox_flat = ox.reshape(-1)  # (grid_size^2,)
            oy_flat = oy.reshape(-1)
            n_samples = ox_flat.shape[0]

            for cam in train_cams:
                wm = weight_map_loader.get(cam, device=device)   # (1, H, W)
                H, W = wm.shape[-2], wm.shape[-1]

                # 拿到每个高斯的屏幕半径 (需要 render 一次, 但用 no_grad)
                render_pkg = render(cam, gaussians, pipe, background,
                                     use_trained_exp=dataset.train_test_exp,
                                     separate_sh=SPARSE_ADAM_AVAILABLE)
                radii = render_pkg["radii"].float()  # (N,) 屏幕空间像素半径

                proj = xyz_h @ cam.full_proj_transform  # (N, 4)
                w_h = proj[:, 3]
                in_front = w_h > 1e-6
                ndc = proj[:, :2] / (w_h.unsqueeze(-1) + 1e-8)
                in_ndc = (ndc[:, 0].abs() < 1) & (ndc[:, 1].abs() < 1)
                on_screen = in_front & in_ndc & (radii > 0)
                if not on_screen.any():
                    continue

                idx = on_screen.nonzero(as_tuple=True)[0]
                # 中心像素坐标
                cx = (ndc[idx, 0] + 1.0) * 0.5 * W  # (M,)
                cy = (ndc[idx, 1] + 1.0) * 0.5 * H
                r = radii[idx].clamp(min=1.0)       # (M,) 至少 1 像素

                # 广播采样: (M, 25)
                sx = (cx.unsqueeze(1) + r.unsqueeze(1) * ox_flat.unsqueeze(0)).long().clamp(0, W - 1)
                sy = (cy.unsqueeze(1) + r.unsqueeze(1) * oy_flat.unsqueeze(0)).long().clamp(0, H - 1)
                w_samples = wm[0, sy, sx]           # (M, 25)

                # 该视角每个高斯的"外溢占比" = 采样点中 wm<thr 的比例
                overflow = (w_samples < thr).float().mean(dim=1)  # (M,)
                # 聚合: 取所有可见视角的 min (最能证明"这个高斯其实落在重点区"的那个视角)
                min_overflow[idx] = torch.minimum(min_overflow[idx], overflow)
                visible_in_any[idx] = True

            # 最终剔除条件: 至少在一个视角可见 & 所有可见视角外溢占比都>阈值 => 剔除
            prune_mask = visible_in_any & (min_overflow > overflow_ratio)
            n_prune = int(prune_mask.sum())
            if n_prune > 0:
                # gaussians.prune_points 依赖 self.tmp_radii, 该字段仅在
                # densify_and_prune 内部临时赋值, 其余时段为 None. 我们在
                # densify 段外调 prune_points 时需临时补一个占位 tmp_radii.
                if gaussians.tmp_radii is None:
                    gaussians.tmp_radii = torch.zeros(N, device=device)
                    _restore_tmp_radii = True
                else:
                    _restore_tmp_radii = False
                gaussians.prune_points(prune_mask)
                if _restore_tmp_radii:
                    gaussians.tmp_radii = None
            print(f"\n[hard-prune @ iter {iteration}] "
                  f"scanned {len(train_cams)} views, "
                  f"N_before={N}, visible_in_any={int(visible_in_any.sum())}, "
                  f"pruned={n_prune}, N_after={gaussians.get_xyz.shape[0]}")
    # ------------------------------------------------------------------

    viewpoint_stack = scene.getTrainCameras().copy()
    viewpoint_indices = list(range(len(viewpoint_stack)))
    ema_loss_for_log = 0.0
    ema_Ll1depth_for_log = 0.0

    progress_bar = tqdm(range(first_iter, opt.iterations), desc="Training progress")
    first_iter += 1
    for iteration in range(first_iter, opt.iterations + 1):
        if network_gui.conn == None:
            network_gui.try_connect()
        while network_gui.conn != None:
            try:
                net_image_bytes = None
                custom_cam, do_training, pipe.convert_SHs_python, pipe.compute_cov3D_python, keep_alive, scaling_modifer = network_gui.receive()
                if custom_cam != None:
                    net_image = render(custom_cam, gaussians, pipe, background, scaling_modifier=scaling_modifer, use_trained_exp=dataset.train_test_exp, separate_sh=SPARSE_ADAM_AVAILABLE)["render"]
                    net_image_bytes = memoryview((torch.clamp(net_image, min=0, max=1.0) * 255).byte().permute(1, 2, 0).contiguous().cpu().numpy())
                network_gui.send(net_image_bytes, dataset.source_path)
                if do_training and ((iteration < int(opt.iterations)) or not keep_alive):
                    break
            except Exception as e:
                network_gui.conn = None

        iter_start.record()

        gaussians.update_learning_rate(iteration)

        # Every 1000 its we increase the levels of SH up to a maximum degree
        if iteration % 1000 == 0:
            gaussians.oneupSHdegree()

        # Pick a random Camera
        if not viewpoint_stack:
            viewpoint_stack = scene.getTrainCameras().copy()
            viewpoint_indices = list(range(len(viewpoint_stack)))
        rand_idx = randint(0, len(viewpoint_indices) - 1)
        viewpoint_cam = viewpoint_stack.pop(rand_idx)
        vind = viewpoint_indices.pop(rand_idx)

        # Render
        if (iteration - 1) == debug_from:
            pipe.debug = True

        bg = torch.rand((3), device="cuda") if opt.random_background else background

        render_pkg = render(viewpoint_cam, gaussians, pipe, bg, use_trained_exp=dataset.train_test_exp, separate_sh=SPARSE_ADAM_AVAILABLE)
        image, viewspace_point_tensor, visibility_filter, radii = render_pkg["render"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]

        if viewpoint_cam.alpha_mask is not None:
            alpha_mask = viewpoint_cam.alpha_mask.cuda()
            image *= alpha_mask

        # Loss
        gt_image = viewpoint_cam.original_image.cuda()
        weight_map = None  # 保留到 densify 段用
        if weight_map_loader is not None:
            weight_map = weight_map_loader.get(viewpoint_cam, device=image.device)
            # 关键一行：仅在 L1 上加权，SSIM 完全不动
            Ll1 = (weight_map * torch.abs(image - gt_image)).mean()
        else:
            Ll1 = l1_loss(image, gt_image)
        if FUSED_SSIM_AVAILABLE:
            ssim_value = fused_ssim(image.unsqueeze(0), gt_image.unsqueeze(0))
        else:
            ssim_value = ssim(image, gt_image)

        loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim_value)

        # -------- 各向异性正则 (07-19_04 / 07-20 迭代, 抑制狭长高斯) --------
        # 两种模式:
        #   hard: aniso_loss = mean(relu(ratio - MAX_RATIO))
        #         只惩罚 ratio > MAX_RATIO 的部分, 缺点是梯度只推到阈值就停
        #         (07-19_04 l1r10 观察到针精准卡在 aniso=8-10 天花板下)
        #   soft: aniso_loss = mean(relu(ratio - 1)) = mean(ratio - 1)
        #         全程有梯度朝球形推 (ratio=1), 更彻底
        # lambda_aniso=0.0 (默认) 关闭正则, 保持完全向后兼容
        Laniso = torch.tensor(0.0, device=image.device)
        if getattr(opt, "lambda_aniso", 0.0) > 0:
            scales = gaussians.get_scaling  # (N, 3) 已 exp 激活后的真实 scale
            max_s = scales.max(dim=1).values
            min_s = scales.min(dim=1).values
            ratio = max_s / (min_s + 1e-8)
            aniso_mode = getattr(opt, "aniso_mode", "hard")
            if aniso_mode == "soft":
                # ratio >= 1 恒成立, 直接 mean(ratio - 1); relu 保安全
                Laniso = torch.mean(torch.clamp(ratio - 1.0, min=0.0))
            else:  # hard
                Laniso = torch.mean(torch.clamp(ratio - opt.aniso_max_ratio, min=0.0))
            loss = loss + opt.lambda_aniso * Laniso
        # -----------------------------------------------------------------

        # Depth regularization
        Ll1depth_pure = 0.0
        if depth_l1_weight(iteration) > 0 and viewpoint_cam.depth_reliable:
            invDepth = render_pkg["depth"]
            mono_invdepth = viewpoint_cam.invdepthmap.cuda()
            depth_mask = viewpoint_cam.depth_mask.cuda()

            Ll1depth_pure = torch.abs((invDepth  - mono_invdepth) * depth_mask).mean()
            Ll1depth = depth_l1_weight(iteration) * Ll1depth_pure 
            loss += Ll1depth
            Ll1depth = Ll1depth.item()
        else:
            Ll1depth = 0

        loss.backward()

        iter_end.record()

        with torch.no_grad():
            # Progress bar
            ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
            ema_Ll1depth_for_log = 0.4 * Ll1depth + 0.6 * ema_Ll1depth_for_log

            if iteration % 10 == 0:
                progress_bar.set_postfix({"Loss": f"{ema_loss_for_log:.{7}f}", "Depth Loss": f"{ema_Ll1depth_for_log:.{7}f}"})
                progress_bar.update(10)
            if iteration == opt.iterations:
                progress_bar.close()

            # Log and save
            training_report(tb_writer, iteration, Ll1, loss, l1_loss, iter_start.elapsed_time(iter_end), testing_iterations, scene, render, (pipe, background, 1., SPARSE_ADAM_AVAILABLE, None, dataset.train_test_exp), dataset.train_test_exp)
            if (iteration in saving_iterations):
                print("\n[ITER {}] Saving Gaussians".format(iteration))
                scene.save(iteration)

            # Densification
            if iteration < opt.densify_until_iter:
                # Keep track of max radii in image-space for pruning
                gaussians.max_radii2D[visibility_filter] = torch.max(gaussians.max_radii2D[visibility_filter], radii[visibility_filter])

                # -------- 权重图硬门控 (07-19_03, 从 fbe0dfc 移植) --------
                # 只有可见 & 投影像素处权重 > thr 的高斯才计入 densify 统计
                # 低权重区 (水体) 的高斯 -> 梯度不累积 -> 达不到 densify_grad_threshold -> 不分裂
                # weight_densify_thr <= 0 时走原路 (loss-only 兼容)
                if (opt.use_weight_map
                        and weight_map is not None
                        and opt.weight_densify_thr > 0):
                    H, W = image.shape[1], image.shape[2]
                    # visibility_filter 兼容: (N,) bool 或 (M,) long -> 统一转 long index
                    if visibility_filter.dtype == torch.bool:
                        vis_idx = visibility_filter.nonzero(as_tuple=True)[0]
                    else:
                        vis_idx = visibility_filter.squeeze(-1) if visibility_filter.dim() > 1 else visibility_filter
                    # 3D 中心 -> NDC -> 像素
                    xyz = gaussians.get_xyz[vis_idx]
                    ones = torch.ones(xyz.shape[0], 1, device=xyz.device)
                    xyz_h = torch.cat([xyz, ones], dim=1)
                    proj = xyz_h @ viewpoint_cam.full_proj_transform
                    ndc = proj[:, :2] / (proj[:, 3:4] + 1e-8)
                    px = ((ndc[:, 0] + 1.0) * 0.5 * W).long().clamp(0, W - 1)
                    py = ((ndc[:, 1] + 1.0) * 0.5 * H).long().clamp(0, H - 1)
                    vis_weights = weight_map[0, py, px]
                    high_conf_mask = vis_weights > opt.weight_densify_thr
                    # 构造与全部高斯同长度的 bool mask, 只标记"可见 且 高权重"的
                    N_all = gaussians.get_xyz.shape[0]
                    high_conf_full = torch.zeros(N_all, dtype=torch.bool, device=xyz.device)
                    if high_conf_mask.any():
                        high_conf_full[vis_idx[high_conf_mask]] = True
                    gaussians.add_densification_stats(viewspace_point_tensor, high_conf_full)
                else:
                    gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)
                # ----------------------------------------------------------

                if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0:
                    size_threshold = 20 if iteration > opt.opacity_reset_interval else None
                    gaussians.densify_and_prune(opt.densify_grad_threshold, 0.005, scene.cameras_extent, size_threshold, radii)
                
                if iteration % opt.opacity_reset_interval == 0 or (dataset.white_background and iteration == opt.densify_from_iter):
                    gaussians.reset_opacity()

            # -------- 07-20_02 · hard prune 触发点 (整个训练全程启用) --------
            # 每 wm_hard_prune_interval iter 扫一次, 物理删除多视角均在非重点区的高斯
            # 放在 densify 段之外, 因为要求整个 30k iter 均启用 (不只是 densify 期 15k)
            if (opt.use_weight_map
                    and weight_map_loader is not None
                    and getattr(opt, "wm_hard_prune_thr", 0.0) > 0
                    and iteration > 0
                    and iteration % opt.wm_hard_prune_interval == 0
                    and iteration < opt.iterations):  # 最后一步不要 prune, 避免破坏保存
                wm_hard_prune_step(gaussians, scene, weight_map_loader,
                                   opt.wm_hard_prune_thr, iteration,
                                   grid_size=getattr(opt, "wm_hard_prune_grid", 5),
                                   radius_mul=getattr(opt, "wm_hard_prune_radius_mul", 2.0),
                                   overflow_ratio=getattr(opt, "wm_hard_prune_overflow_ratio", 0.7),
                                   pipe=pipe, background=background)
            # ---------------------------------------------------------------

            # Optimizer step
            if iteration < opt.iterations:
                gaussians.exposure_optimizer.step()
                gaussians.exposure_optimizer.zero_grad(set_to_none = True)
                if use_sparse_adam:
                    visible = radii > 0
                    gaussians.optimizer.step(visible, radii.shape[0])
                    gaussians.optimizer.zero_grad(set_to_none = True)
                else:
                    gaussians.optimizer.step()
                    gaussians.optimizer.zero_grad(set_to_none = True)

            if (iteration in checkpoint_iterations):
                print("\n[ITER {}] Saving Checkpoint".format(iteration))
                torch.save((gaussians.capture(), iteration), scene.model_path + "/chkpnt" + str(iteration) + ".pth")

def prepare_output_and_logger(args):    
    if not args.model_path:
        if os.getenv('OAR_JOB_ID'):
            unique_str=os.getenv('OAR_JOB_ID')
        else:
            unique_str = str(uuid.uuid4())
        args.model_path = os.path.join("./output/", unique_str[0:10])
        
    # Set up output folder
    print("Output folder: {}".format(args.model_path))
    os.makedirs(args.model_path, exist_ok = True)
    with open(os.path.join(args.model_path, "cfg_args"), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))

    # Create Tensorboard writer
    tb_writer = None
    if TENSORBOARD_FOUND:
        tb_writer = SummaryWriter(args.model_path)
    else:
        print("Tensorboard not available: not logging progress")
    return tb_writer

def training_report(tb_writer, iteration, Ll1, loss, l1_loss, elapsed, testing_iterations, scene : Scene, renderFunc, renderArgs, train_test_exp):
    if tb_writer:
        tb_writer.add_scalar('train_loss_patches/l1_loss', Ll1.item(), iteration)
        tb_writer.add_scalar('train_loss_patches/total_loss', loss.item(), iteration)
        tb_writer.add_scalar('iter_time', elapsed, iteration)

    # Report test and samples of training set
    if iteration in testing_iterations:
        torch.cuda.empty_cache()
        validation_configs = ({'name': 'test', 'cameras' : scene.getTestCameras()}, 
                              {'name': 'train', 'cameras' : [scene.getTrainCameras()[idx % len(scene.getTrainCameras())] for idx in range(5, 30, 5)]})

        for config in validation_configs:
            if config['cameras'] and len(config['cameras']) > 0:
                l1_test = 0.0
                psnr_test = 0.0
                for idx, viewpoint in enumerate(config['cameras']):
                    image = torch.clamp(renderFunc(viewpoint, scene.gaussians, *renderArgs)["render"], 0.0, 1.0)
                    gt_image = torch.clamp(viewpoint.original_image.to("cuda"), 0.0, 1.0)
                    if train_test_exp:
                        image = image[..., image.shape[-1] // 2:]
                        gt_image = gt_image[..., gt_image.shape[-1] // 2:]
                    if tb_writer and (idx < 5):
                        tb_writer.add_images(config['name'] + "_view_{}/render".format(viewpoint.image_name), image[None], global_step=iteration)
                        if iteration == testing_iterations[0]:
                            tb_writer.add_images(config['name'] + "_view_{}/ground_truth".format(viewpoint.image_name), gt_image[None], global_step=iteration)
                    l1_test += l1_loss(image, gt_image).mean().double()
                    psnr_test += psnr(image, gt_image).mean().double()
                psnr_test /= len(config['cameras'])
                l1_test /= len(config['cameras'])          
                print("\n[ITER {}] Evaluating {}: L1 {} PSNR {}".format(iteration, config['name'], l1_test, psnr_test))
                if tb_writer:
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - l1_loss', l1_test, iteration)
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - psnr', psnr_test, iteration)

        if tb_writer:
            tb_writer.add_histogram("scene/opacity_histogram", scene.gaussians.get_opacity, iteration)
            tb_writer.add_scalar('total_points', scene.gaussians.get_xyz.shape[0], iteration)
        torch.cuda.empty_cache()

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument('--ip', type=str, default="127.0.0.1")
    parser.add_argument('--port', type=int, default=6009)
    parser.add_argument('--debug_from', type=int, default=-1)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[7_000, 30_000])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[7_000, 30_000])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument('--disable_viewer', action='store_true', default=False)
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--start_checkpoint", type=str, default = None)
    args = parser.parse_args(sys.argv[1:])
    args.save_iterations.append(args.iterations)
    
    print("Optimizing " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    # Start GUI server, configure and run training
    if not args.disable_viewer:
        network_gui.init(args.ip, args.port)
    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    training(lp.extract(args), op.extract(args), pp.extract(args), args.test_iterations, args.save_iterations, args.checkpoint_iterations, args.start_checkpoint, args.debug_from)

    # All done
    print("\nTraining complete.")
