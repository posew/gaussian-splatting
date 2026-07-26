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
import torch.nn.functional as F
from random import randint
from utils.loss_utils import l1_loss, ssim
from gaussian_renderer import render, network_gui
import sys
from scene import Scene, GaussianModel
from utils.general_utils import safe_state, get_expon_lr_func, inverse_sigmoid
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
            method=getattr(opt, "weight_map_method", "legacy"),
            sfm_kde_sigma_px=getattr(opt, "sfm_kde_sigma_px", 25.0),
            sfm_kde_floor=getattr(opt, "sfm_kde_floor", 0.05),
            locvar_dist_edge_thr=getattr(opt, "locvar_dist_edge_thr", 0.4),
            locvar_dist_sigma_ratio=getattr(opt, "locvar_dist_sigma_ratio", 1.0/15),
            locvar_dist_floor=getattr(opt, "locvar_dist_floor", 0.05),
            wm_v2_kmeans_k=getattr(opt, "wm_v2_kmeans_k", 16),
            wm_v2_caustic_L=getattr(opt, "wm_v2_caustic_L", 200),
            wm_v2_caustic_chroma=getattr(opt, "wm_v2_caustic_chroma", 40),
            wm_v2_caustic_dilate=getattr(opt, "wm_v2_caustic_dilate", 9),
            wm_v2_colmap_seed_radius=getattr(opt, "wm_v2_colmap_seed_radius", 15),
            wm_v2_use_colmap_seed=getattr(opt, "wm_v2_use_colmap_seed", True),
        )
        print(
            f"[weightmap] Enabled: method={getattr(opt,'weight_map_method','legacy')}, "
            f"mode={opt.weight_map_mode}, "
            f"alpha={opt.weight_map_alpha}, beta={opt.weight_map_beta}, "
            f"densify_thr={opt.weight_densify_thr}"
            + (" (loss-only)" if opt.weight_densify_thr <= 0 else " (loss + hard densify gate)")
        )

    viewpoint_stack = scene.getTrainCameras().copy()
    viewpoint_indices = list(range(len(viewpoint_stack)))
    ema_loss_for_log = 0.0
    ema_Ll1depth_for_log = 0.0

    # M3: 预构建腐蚀核 (用于 SSIM mask 边界腐蚀)
    use_hard_mask = (weight_map_loader is not None and opt.wm_hard_thr > 0
                     and getattr(opt, "weight_map_method", "legacy") == "wm_v2")
    if use_hard_mask:
        erode_k = max(3, opt.wm_erode_px * 2 + 1)
        erode_kernel = torch.ones(1, 1, erode_k, erode_k, device="cuda")
        print(f"[M3] Hard mask ON: thr={opt.wm_hard_thr}, erode={opt.wm_erode_px}px, "
              f"explicit_bg={opt.use_explicit_bg}, "
              f"prune_interval={opt.wm_prune_interval}, prune_decay={opt.wm_prune_decay}")

    # M3.3: prune gating 累积器 (多视角 wm 平均)
    wm_accum = None
    wm_accum_count = None

    # -------- M4: 水下介质模型 (C1) --------
    medium_model = None
    medium_optimizer = None
    if getattr(opt, "use_medium", False):
        from utils.medium_model import MediumModel
        medium_model = MediumModel(
            beta_init=(opt.beta_init_r, opt.beta_init_g, opt.beta_init_b),
        ).cuda()
        medium_optimizer = torch.optim.Adam(medium_model.parameters(), lr=opt.medium_lr)
        # B_inf 初始化延迟到第一帧拿到 mask 后 (init_B_from_bg_pixels)
        medium_B_initialized = False
        print(f"[M4] Medium model ON: β_init=({opt.beta_init_r},{opt.beta_init_g},{opt.beta_init_b}), "
              f"warmup={opt.medium_warmup_iter}, β_free={opt.medium_beta_free_iter}, "
              f"w_mono={opt.w_mono}, lr={opt.medium_lr}")
    # ----------------------------------------

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
        hard_mask = None   # M3: 二值前景 mask

        if weight_map_loader is not None:
            weight_map = weight_map_loader.get(viewpoint_cam, device=image.device)

        if use_hard_mask and weight_map is not None:
            # ---- M3.1: hard mask + caustic + erode ----
            caustic = weight_map_loader.get_caustic_mask(viewpoint_cam, device=image.device)
            hard_mask = (weight_map > opt.wm_hard_thr).float()
            if caustic is not None:
                hard_mask = hard_mask * (1.0 - caustic)  # 光斑区置 0

            # ---- M4: 介质模型 compositing (替代 M3.4 显式背景) ----
            if medium_model is not None and iteration > opt.medium_warmup_iter:
                # B_inf 首帧初始化
                if not medium_B_initialized:
                    medium_model.init_B_from_bg_pixels(gt_image, hard_mask)
                    medium_B_initialized = True
                    print(f"[M4] B_inf initialized to {medium_model.B_inf.data.tolist()}")

                # 冻结 β (warmup < iter < beta_free) 或全部可学
                beta_frozen = iteration < opt.medium_beta_free_iter
                if beta_frozen:
                    medium_model.beta_raw.requires_grad_(False)
                else:
                    medium_model.beta_raw.requires_grad_(True)

                depth_map = render_pkg["depth"]  # (1, H, W)
                J = image  # 高斯渲染的 radiance
                I_fg = medium_model(J, depth_map)
                B_medium = medium_model.B_inf.view(3, 1, 1)
                image = hard_mask * I_fg + (1.0 - hard_mask) * B_medium

            # M3.4: 显式背景色 compositing (M4 关闭时 fallback)
            elif opt.use_explicit_bg:
                bg_mask_bool = (hard_mask < 0.5)  # (1, H, W) bool
                bg_medians = []
                for c in range(3):
                    ch_bg = gt_image[c][bg_mask_bool[0]]
                    if ch_bg.numel() > 0:
                        bg_medians.append(ch_bg.median())
                    else:
                        bg_medians.append(torch.tensor(0.0, device=image.device))
                B_view = torch.stack(bg_medians).view(3, 1, 1)  # (3, 1, 1)
                image = hard_mask * image + (1.0 - hard_mask) * B_view

            # L1: masked mean
            Ll1 = (hard_mask * torch.abs(image - gt_image)).sum() / (hard_mask.sum() * 3 + 1e-6)

            # SSIM: eroded mask (避免边界伪影)
            mask_4d = hard_mask.unsqueeze(0)  # (1, 1, H, W)
            mask_erode = F.conv2d(mask_4d, erode_kernel, padding=erode_k // 2)
            mask_erode = (mask_erode >= erode_kernel.numel()).float().squeeze(0)  # (1, H, W)
            image_masked = image * mask_erode
            gt_masked = gt_image * mask_erode
            if FUSED_SSIM_AVAILABLE:
                ssim_value = fused_ssim(image_masked.unsqueeze(0), gt_masked.unsqueeze(0))
            else:
                ssim_value = ssim(image_masked, gt_masked)
        else:
            # 原始路径: soft weight map or no weight map
            if weight_map is not None:
                Ll1 = (weight_map * torch.abs(image - gt_image)).mean()
            else:
                Ll1 = l1_loss(image, gt_image)
            if FUSED_SSIM_AVAILABLE:
                ssim_value = fused_ssim(image.unsqueeze(0), gt_image.unsqueeze(0))
            else:
                ssim_value = ssim(image, gt_image)

        loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim_value)

        # Depth regularization
        Ll1depth_pure = 0.0
        if depth_l1_weight(iteration) > 0 and viewpoint_cam.depth_reliable:
            invDepth = render_pkg["depth"]
            mono_invdepth = viewpoint_cam.invdepthmap.cuda()
            depth_mask = viewpoint_cam.depth_mask.cuda()

            if hard_mask is not None:
                depth_mask = depth_mask * hard_mask  # M3: 前景区域才算 depth loss

            Ll1depth_pure = torch.abs((invDepth  - mono_invdepth) * depth_mask).mean()
            Ll1depth = depth_l1_weight(iteration) * Ll1depth_pure
            loss += Ll1depth
            Ll1depth = Ll1depth.item()
        else:
            Ll1depth = 0

        # M4: 通道单调约束 (仅 β 可学阶段)
        if (medium_model is not None
                and iteration > opt.medium_beta_free_iter):
            L_mono = medium_model.mono_loss()
            loss = loss + opt.w_mono * L_mono
        else:
            L_mono = None

        loss.backward()

        # M4: medium optimizer step (独立于高斯 optimizer)
        if medium_model is not None and iteration > opt.medium_warmup_iter:
            medium_optimizer.step()
            medium_optimizer.zero_grad(set_to_none=True)
            if tb_writer:
                tb_writer.add_scalar('medium/beta_R', medium_model.beta[0].item(), iteration)
                tb_writer.add_scalar('medium/beta_G', medium_model.beta[1].item(), iteration)
                tb_writer.add_scalar('medium/beta_B', medium_model.beta[2].item(), iteration)
                tb_writer.add_scalar('medium/B_inf_R', medium_model.B_inf[0].item(), iteration)
                tb_writer.add_scalar('medium/B_inf_G', medium_model.B_inf[1].item(), iteration)
                tb_writer.add_scalar('medium/B_inf_B', medium_model.B_inf[2].item(), iteration)
                if L_mono is not None:
                    tb_writer.add_scalar('medium/L_mono', L_mono.item(), iteration)
            if iteration % 5000 == 0:
                b = medium_model.beta.data.tolist()
                bi = medium_model.B_inf.data.tolist()
                mono_str = f", L_mono={L_mono.item():.4f}" if L_mono is not None else ""
                print(f"[M4] iter {iteration}: β=({b[0]:.3f},{b[1]:.3f},{b[2]:.3f}), "
                      f"B∞=({bi[0]:.3f},{bi[1]:.3f},{bi[2]:.3f}){mono_str}")

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

            # -------- M3.3: Prune 门控 (bg 高斯 opacity 半衰) --------
            # MUST run before densification — densify_and_prune changes gaussian
            # indices, making the current visibility_filter stale.
            if (use_hard_mask
                    and opt.wm_prune_interval > 0
                    and weight_map is not None):
                N_all = gaussians.get_xyz.shape[0]
                if wm_accum is None or wm_accum.shape[0] != N_all:
                    wm_accum = torch.zeros(N_all, device="cuda")
                    wm_accum_count = torch.zeros(N_all, device="cuda")
                if visibility_filter.dtype == torch.bool:
                    vis_idx = visibility_filter.nonzero(as_tuple=True)[0]
                else:
                    vis_idx = visibility_filter.squeeze(-1) if visibility_filter.dim() > 1 else visibility_filter
                H_img, W_img = image.shape[1], image.shape[2]
                xyz_vis = gaussians.get_xyz[vis_idx]
                ones_vis = torch.ones(xyz_vis.shape[0], 1, device=xyz_vis.device)
                xyz_h = torch.cat([xyz_vis, ones_vis], dim=1)
                proj = xyz_h @ viewpoint_cam.full_proj_transform
                ndc = proj[:, :2] / (proj[:, 3:4] + 1e-8)
                px = ((ndc[:, 0] + 1.0) * 0.5 * W_img).long().clamp(0, W_img - 1)
                py = ((ndc[:, 1] + 1.0) * 0.5 * H_img).long().clamp(0, H_img - 1)
                vis_wm = weight_map[0, py, px]
                wm_accum[vis_idx] += vis_wm
                wm_accum_count[vis_idx] += 1.0

                if iteration % opt.wm_prune_interval == 0 and iteration > opt.densify_from_iter:
                    observed = wm_accum_count > 0
                    avg_wm = torch.zeros(N_all, device="cuda")
                    avg_wm[observed] = wm_accum[observed] / wm_accum_count[observed]
                    bg_gaussians = observed & (avg_wm < opt.wm_hard_thr)
                    if bg_gaussians.any():
                        cur_opacity = gaussians.get_opacity.squeeze(-1)
                        decayed = cur_opacity.clone()
                        decayed[bg_gaussians] *= opt.wm_prune_decay
                        new_opacity_logit = inverse_sigmoid(decayed.clamp(1e-4, 1 - 1e-4))
                        gaussians._opacity.data[:, 0] = new_opacity_logit
                        n_bg = int(bg_gaussians.sum().item())
                        if iteration % 2000 == 0:
                            print(f"[M3.3 prune] iter {iteration}: {n_bg}/{N_all} bg gaussians decayed (×{opt.wm_prune_decay})")
                    wm_accum.zero_()
                    wm_accum_count.zero_()
            # -------------------------------------------------------

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
