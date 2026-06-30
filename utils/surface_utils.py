import torch
import torch.nn.functional as F
from utils.general_utils import build_rotation


def compute_normal_consistency_loss(
    gaussians,
    weight_scores=None,
    k_neighbors: int = 10,
    lambda_normal: float = 0.01,
    sample_size: int = 2000,
) -> torch.Tensor:
    xyz = gaussians.get_xyz
    scales = gaussians.get_scaling
    rots = gaussians.get_rotation

    rot_mats = build_rotation(rots)

    min_axis_idx = scales.argmin(dim=-1)
    N = xyz.shape[0]
    normals = rot_mats[torch.arange(N, device=xyz.device), :, min_axis_idx]
    normals = F.normalize(normals, dim=-1)

    if N < k_neighbors + 1:
        return torch.tensor(0.0, device=xyz.device, requires_grad=False)

    s = min(N, sample_size)
    sample_idx = torch.randperm(N, device=xyz.device)[:s]
    xyz_s = xyz[sample_idx]
    normals_s = normals[sample_idx]

    dist = torch.cdist(xyz_s, xyz_s)
    dist.fill_diagonal_(float('inf'))
    _, nn_idx = dist.topk(k_neighbors, dim=-1, largest=False)

    nn_normals = normals_s[nn_idx]
    center_normals = normals_s.unsqueeze(1).expand_as(nn_normals)
    cos_sim = (center_normals * nn_normals).sum(dim=-1).abs()
    normal_loss_per = (1.0 - cos_sim).mean(dim=-1)

    if weight_scores is not None:
        w = weight_scores[sample_idx].clamp(0, 1).detach()
        weighted_loss = (w * normal_loss_per).mean()
    else:
        weighted_loss = normal_loss_per.mean()

    return lambda_normal * weighted_loss
