#
# train_weighted.py - 水下图像可信度权重图引导的 3DGS 训练脚本
#
# 基于原始 train.py，在以下位置添加了权重图逻辑：
#   1. Loss 加权：逐像素 L1 loss 乘以可信度权重图
#   2. Densification 抑制：低权重区域的 Gaussian 不参与分裂
#
# 原始 train.py 完全不动，两个脚本可以分别跑对比实验：
#   python train.py           -s <data> -m output/baseline
#   python train_weighted.py  -s <data> -m output/weighted --use_weight_map
#

import os
import torch
from random import randint
from utils.loss_utils import l1_loss, ssim
from gaussian_renderer import render, network_gui
import sys
from scene import Scene, GaussianModel
from utils.general_utils import safe_state, get_expon_lr_func
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

from utils.weight_map_utils import WeightMapLoader
from utils.surface_utils import compute_normal_consistency_loss


def training(dataset, opt, pipe, testing_iterations, saving_iterations,
             checkpoint_iterations, checkpoint, debug_from):

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

    iter_start = torch.cuda.Event(enable_timing=True)
    iter_end = torch.cuda.Event(enable_timing=True)

    use_sparse_adam = opt.optimizer_type == "sparse_adam" and SPARSE_ADAM_AVAILABLE
    depth_l1_weight = get_expon_lr_func(opt.depth_l1_weight_init, opt.depth_l1_weight_final, max_steps=opt.iterations)

    viewpoint_stack = scene.getTrainCameras().copy()
    viewpoint_indices = list(range(len(viewpoint_stack)))
    ema_loss_for_log = 0.0
    ema_Ll1depth_for_log = 0.0

    # ── 初始化权重图加载器 ─────────────────────────────────────────────
    use_weight_map = getattr(opt, "use_weight_map", False)
    weight_loader = None
    weight_densify_thr = getattr(opt, "weight_densify_thr", 0.3)
    if use_weight_map:
        weight_loader = WeightMapLoader(
            source_path=dataset.source_path,
            mode=getattr(opt, "weight_map_mode", "online"),
            alpha=getattr(opt, "weight_map_alpha", 1.0),
            beta=getattr(opt, "weight_map_beta", 1.0),
            weight_map_dir=getattr(opt, "weight_map_dir", "weight_maps"),
        )
        print(f"[WeightMap] Enabled: mode={weight_loader.mode}, "
              f"alpha={weight_loader.alpha}, beta={weight_loader.beta}, "
              f"densify_thr={weight_densify_thr}")
    else:
        print("[WeightMap] Disabled: running as vanilla 3DGS")
    # ──────────────────────────────────────────────────────────────────

    # 贡献度近似：累计每个高斯自上次剪枝以来被观测到（可见）的帧数
    gaussian_visibility_count = None

    progress_bar = tqdm(range(first_iter, opt.iterations), desc="Training progress")
    first_iter += 1
    for iteration in range(first_iter, opt.iterations + 1):
        if network_gui.conn is None:
            network_gui.try_connect()
        while network_gui.conn is not None:
            try:
                net_image_bytes = None
                custom_cam, do_training, pipe.convert_SHs_python, pipe.compute_cov3D_python, keep_alive, scaling_modifer = network_gui.receive()
                if custom_cam is not None:
                    net_image = render(custom_cam, gaussians, pipe, background, scaling_modifier=scaling_modifer, use_trained_exp=dataset.train_test_exp, separate_sh=SPARSE_ADAM_AVAILABLE)["render"]
                    net_image_bytes = memoryview((torch.clamp(net_image, min=0, max=1.0) * 255).byte().permute(1, 2, 0).contiguous().cpu().numpy())
                network_gui.send(net_image_bytes, dataset.source_path)
                if do_training and ((iteration < int(opt.iterations)) or not keep_alive):
                    break
            except Exception as e:
                network_gui.conn = None

        iter_start.record()
        gaussians.update_learning_rate(iteration)

        if iteration % 1000 == 0:
            gaussians.oneupSHdegree()

        if not viewpoint_stack:
            viewpoint_stack = scene.getTrainCameras().copy()
            viewpoint_indices = list(range(len(viewpoint_stack)))
        rand_idx = randint(0, len(viewpoint_indices) - 1)
        viewpoint_cam = viewpoint_stack.pop(rand_idx)
        vind = viewpoint_indices.pop(rand_idx)

        if (iteration - 1) == debug_from:
            pipe.debug = True

        bg = torch.rand((3), device="cuda") if opt.random_background else background

        render_pkg = render(viewpoint_cam, gaussians, pipe, bg,
                            use_trained_exp=dataset.train_test_exp,
                            separate_sh=SPARSE_ADAM_AVAILABLE)
        image, viewspace_point_tensor, visibility_filter, radii = (
            render_pkg["render"],
            render_pkg["viewspace_points"],
            render_pkg["visibility_filter"],
            render_pkg["radii"],
        )

        if viewpoint_cam.alpha_mask is not None:
            alpha_mask = viewpoint_cam.alpha_mask.cuda()
            image *= alpha_mask

        gt_image = viewpoint_cam.original_image.cuda()

        # ── Loss 计算 ──────────────────────────────────────────────────
        weight_map = None
        if use_weight_map and weight_loader is not None:
            weight_map = weight_loader.get(viewpoint_cam, device="cuda")  # (1, H, W)
            if weight_map is not None:
                H, W = image.shape[1], image.shape[2]
                if weight_map.shape[1] != H or weight_map.shape[2] != W:
                    import torch.nn.functional as F
                    weight_map = F.interpolate(
                        weight_map.unsqueeze(0), size=(H, W),
                        mode="bilinear", align_corners=False
                    ).squeeze(0)
                # 加权 L1：高可信度区域贡献更大梯度
                Ll1 = (weight_map * torch.abs(image - gt_image)).mean()
            else:
                Ll1 = l1_loss(image, gt_image)
        else:
            Ll1 = l1_loss(image, gt_image)  # 原始 3DGS 行为

        if FUSED_SSIM_AVAILABLE:
            ssim_value = fused_ssim(image.unsqueeze(0), gt_image.unsqueeze(0))
        else:
            ssim_value = ssim(image, gt_image)

        loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim_value)
        # ──────────────────────────────────────────────────────────────

        # Depth regularization（与原始 train.py 完全一致）
        Ll1depth_pure = 0.0
        if depth_l1_weight(iteration) > 0 and viewpoint_cam.depth_reliable:
            invDepth = render_pkg["depth"]
            mono_invdepth = viewpoint_cam.invdepthmap.cuda()
            depth_mask = viewpoint_cam.depth_mask.cuda()
            Ll1depth_pure = torch.abs((invDepth - mono_invdepth) * depth_mask).mean()
            Ll1depth = depth_l1_weight(iteration) * Ll1depth_pure
            loss += Ll1depth
            Ll1depth = Ll1depth.item()
        else:
            Ll1depth = 0

        # ── 可见频次积累：贡献度近似（每帧对可见的高斯计数 +1）─────────────
        if iteration < opt.densify_until_iter:
            n_gaussians = gaussians.get_xyz.shape[0]
            if gaussian_visibility_count is None or gaussian_visibility_count.shape[0] != n_gaussians:
                gaussian_visibility_count = torch.zeros(n_gaussians, device="cuda")
            gaussian_visibility_count[visibility_filter] += 1
        # ──────────────────────────────────────────────────────────────

        # ── 法向一致性约束（模块3）────────────────────────────────────
        if use_weight_map and getattr(opt, "use_normal_loss", False):
            normal_loss = compute_normal_consistency_loss(
                gaussians,
                weight_scores=None,
                k_neighbors=10,
                lambda_normal=getattr(opt, "lambda_normal", 0.01),
                sample_size=2000,
            )
            loss = loss + normal_loss
        # ──────────────────────────────────────────────────────────────

        loss.backward()
        iter_end.record()

        with torch.no_grad():
            ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
            ema_Ll1depth_for_log = 0.4 * Ll1depth + 0.6 * ema_Ll1depth_for_log

            if iteration % 10 == 0:
                progress_bar.set_postfix({
                    "Loss": f"{ema_loss_for_log:.{7}f}",
                    "Depth Loss": f"{ema_Ll1depth_for_log:.{7}f}",
                })
                progress_bar.update(10)
            if iteration == opt.iterations:
                progress_bar.close()

            training_report(tb_writer, iteration, Ll1, loss, l1_loss,
                            iter_start.elapsed_time(iter_end), testing_iterations,
                            scene, render,
                            (pipe, background, 1., SPARSE_ADAM_AVAILABLE, None, dataset.train_test_exp),
                            dataset.train_test_exp)

            if iteration in saving_iterations:
                print(f"\n[ITER {iteration}] Saving Gaussians")
                scene.save(iteration)

            # ── Densification（带权重图抑制）────────────────────────────
            if iteration < opt.densify_until_iter:
                gaussians.max_radii2D[visibility_filter] = torch.max(
                    gaussians.max_radii2D[visibility_filter],
                    radii[visibility_filter],
                )

                if use_weight_map and weight_map is not None and weight_densify_thr > 0:
                    H, W = image.shape[1], image.shape[2]
                    # visibility_filter 是 (M,1) 的 indices，展平为 1D
                    vis_idx = visibility_filter.squeeze(-1)
                    # 用相机投影矩阵将 3D 高斯中心投影到 2D 像素坐标
                    xyz = gaussians.get_xyz[vis_idx]  # (M, 3)
                    ones = torch.ones(xyz.shape[0], 1, device="cuda")
                    xyz_h = torch.cat([xyz, ones], dim=1)  # (M, 4)
                    # full_proj_transform: world -> clip space (4x4)
                    proj = xyz_h @ viewpoint_cam.full_proj_transform  # (M, 4)
                    ndc = proj[:, :2] / (proj[:, 3:4] + 1e-8)  # (M, 2) NDC [-1, 1]
                    px = ((ndc[:, 0] + 1.0) * 0.5 * W).long().clamp(0, W - 1)
                    py = ((ndc[:, 1] + 1.0) * 0.5 * H).long().clamp(0, H - 1)
                    vis_weights = weight_map[0, py, px]
                    high_conf_mask = vis_weights > weight_densify_thr
                    high_conf_full = torch.zeros(
                        viewspace_point_tensor.shape[0], dtype=torch.bool, device="cuda"
                    )
                    if high_conf_mask.any():
                        high_conf_full[vis_idx[high_conf_mask]] = True
                    gaussians.add_densification_stats(viewspace_point_tensor, high_conf_full)
                else:
                    gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)

                if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0:
                    size_threshold = 20 if iteration > opt.opacity_reset_interval else None

                    # ── 新剪枝：几何异常 + 贡献度近似（取代权重图剪枝）──
                    # 贡献度近似用累计可见频次 visibility_count（旧点在前 n_old）。
                    visibility_count = None
                    if (gaussian_visibility_count is not None
                            and gaussian_visibility_count.shape[0] == gaussians.get_xyz.shape[0]):
                        visibility_count = gaussian_visibility_count

                    gaussians.densify_and_prune(
                        opt.densify_grad_threshold, 0.005,
                        scene.cameras_extent, size_threshold,
                        radii,
                        aniso_thr=getattr(opt, "aniso_thr", 8.0),
                        aniso_min_scale_ratio=getattr(opt, "aniso_min_scale_ratio", 0.001),
                        contrib_prune_thr=getattr(opt, "contrib_prune_thr", 0.1),
                        visibility_count=visibility_count,
                    )
                    # 剪枝后可见频次计数失效（点集变了），重置
                    gaussian_visibility_count = None

                if iteration % opt.opacity_reset_interval == 0 or (
                    dataset.white_background and iteration == opt.densify_from_iter
                ):
                    gaussians.reset_opacity()
            # ──────────────────────────────────────────────────────────

            if use_sparse_adam:
                visible = radii > 0
                gaussians.optimizer.step(visible, radii.shape[0])
                gaussians.optimizer.zero_grad(set_to_none=True)
            else:
                gaussians.optimizer.step()
                gaussians.optimizer.zero_grad(set_to_none=True)

            if iteration in checkpoint_iterations:
                print(f"\n[ITER {iteration}] Saving Checkpoint")
                torch.save(
                    (gaussians.capture(), iteration),
                    scene.model_path + "/chkpnt" + str(iteration) + ".pth",
                )


def prepare_output_and_logger(args):
    if not args.model_path:
        if os.getenv('OAR_JOB_ID'):
            unique_str = os.getenv('OAR_JOB_ID')
        else:
            unique_str = str(uuid.uuid4())
        args.model_path = os.path.join("./output/", unique_str[0:10])

    print(f"Output folder: {args.model_path}")
    os.makedirs(args.model_path, exist_ok=True)
    with open(os.path.join(args.model_path, "cfg_args"), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))

    tb_writer = None
    if TENSORBOARD_FOUND:
        tb_writer = SummaryWriter(args.model_path)
    else:
        print("Tensorboard not available: not logging progress")
    return tb_writer


def training_report(tb_writer, iteration, Ll1, loss, l1_loss, elapsed,
                    testing_iterations, scene, renderFunc, renderArgs, train_test_exp):
    if tb_writer:
        tb_writer.add_scalar('train_loss_patches/l1_loss', Ll1.item(), iteration)
        tb_writer.add_scalar('train_loss_patches/total_loss', loss.item(), iteration)
        tb_writer.add_scalar('iter_time', elapsed, iteration)

    if iteration in testing_iterations:
        torch.cuda.empty_cache()
        validation_configs = (
            {'name': 'test', 'cameras': scene.getTestCameras()},
            {'name': 'train', 'cameras': [scene.getTrainCameras()[idx % len(scene.getTrainCameras())] for idx in range(5, 30, 5)]},
        )
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
                    if tb_writer and idx < 5:
                        tb_writer.add_images(config['name'] + f"_view_{viewpoint.image_name}/render", image[None], global_step=iteration)
                        if iteration == testing_iterations[0]:
                            tb_writer.add_images(config['name'] + f"_view_{viewpoint.image_name}/ground_truth", gt_image[None], global_step=iteration)
                    l1_test += l1_loss(image, gt_image).mean().double()
                    psnr_test += psnr(image, gt_image).mean().double()
                psnr_test /= len(config['cameras'])
                l1_test /= len(config['cameras'])
                print(f"\n[ITER {iteration}] Evaluating {config['name']}: L1 {l1_test:.4f} PSNR {psnr_test:.4f}")
                if tb_writer:
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - l1_loss', l1_test, iteration)
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - psnr', psnr_test, iteration)

        if tb_writer:
            tb_writer.add_histogram("scene/opacity_histogram", scene.gaussians.get_opacity, iteration)
            tb_writer.add_scalar('total_points', scene.gaussians.get_xyz.shape[0], iteration)
        torch.cuda.empty_cache()


if __name__ == "__main__":
    parser = ArgumentParser(description="Weighted training script - Underwater 3DGS")
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
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--start_checkpoint", type=str, default=None)

    # 权重图参数
    parser.add_argument("--use_weight_map", action="store_true", default=False,
                        help="启用水下图像可信度权重图引导训练")
    parser.add_argument("--weight_map_mode", type=str, default="online",
                        choices=["online", "precomputed"],
                        help="online=实时计算, precomputed=预计算加载")
    parser.add_argument("--weight_map_alpha", type=float, default=1.0,
                        help="清晰度权重指数 alpha")
    parser.add_argument("--weight_map_beta", type=float, default=1.0,
                        help="传输率权重指数 beta")
    parser.add_argument("--weight_map_dir", type=str, default="weight_maps",
                        help="预计算权重图目录名（位于 source_path 下）")
    parser.add_argument("--weight_densify_thr", type=float, default=0.3,
                        help="低于此权重的区域抑制 densification（0=不抑制）")
    # ── 新剪枝参数 ──
    parser.add_argument("--aniso_thr", type=float, default=8.0,
                        help="各向异性比阈值(max/min scale), 超过则剪枝狭长 floater(0=关闭)")
    parser.add_argument("--aniso_min_scale_ratio", type=float, default=0.001,
                        help="各向异性剪枝的最小尺度保护(相对 scene_extent), 避免误伤细节点")
    parser.add_argument("--contrib_prune_thr", type=float, default=0.1,
                        help="贡献度剪枝比例(百分位), 剪掉可见频次×不透明度最低的比例(0=关闭)")

    args = parser.parse_args(sys.argv[1:])
    args.save_iterations.append(args.iterations)

    print(f"Optimizing {args.model_path}")
    safe_state(args.quiet)

    op_args = op.extract(args)
    op_args.use_weight_map = args.use_weight_map
    op_args.weight_map_mode = args.weight_map_mode
    op_args.weight_map_alpha = args.weight_map_alpha
    op_args.weight_map_beta = args.weight_map_beta
    op_args.weight_map_dir = args.weight_map_dir
    op_args.weight_densify_thr = args.weight_densify_thr
    op_args.aniso_thr = args.aniso_thr
    op_args.aniso_min_scale_ratio = args.aniso_min_scale_ratio
    op_args.contrib_prune_thr = args.contrib_prune_thr

    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    training(
        lp.extract(args), op_args, pp.extract(args),
        args.test_iterations, args.save_iterations,
        args.checkpoint_iterations, args.start_checkpoint,
        args.debug_from,
    )
    print("\nTraining complete.")
