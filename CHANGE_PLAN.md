# 水下3DGS改进计划 - 更改计划文档

> 说明：本文档描述所有改动意图和逻辑，不直接修改代码。
> 请用本地LLM按照此计划逐步实施。
>
> 目标路径：`/Users/yancanfeng/codeFile/ky/gaussian-splatting/CHANGE_PLAN.md`

---

## 总体目标

在现有 `train_weighted.py` + 权重图框架基础上，新增两个模块：
1. **模块2**：权重图引导的自适应Pruning（改进伪影剔除）
2. **模块3**：权重图调制的法向一致性约束（改进表面平滑度）

---

## 模块2：自适应Pruning改进

### 涉及文件
- `scene/gaussian_model.py`（修改）
- `train_weighted.py`（修改）
- `arguments/__init__.py`（修改，添加新参数）

---

### 2.1 `scene/gaussian_model.py`

**改动位置：** `densify_and_prune` 方法（约第452行）

**改动意图：**
给 `densify_and_prune` 增加两个可选参数：
- `weight_scores`：shape为 `[N]` 的 float32 Tensor，每个值是对应Gaussian的平均可信度分数（来自权重图），值域 `[0, 1]`
- `weight_prune_thr`：float，低于此阈值的Gaussian额外加入prune_mask

**改动逻辑：**
在原有的 `prune_mask`（opacity低 + 尺寸大）计算完成后，追加新条件：
- 如果 `weight_scores` 不为 None 且 `weight_prune_thr > 0`：
  - `densify_and_clone` 和 `densify_and_split` 执行后，Gaussian总数 `N_current` 会增加
  - `weight_scores` 的长度是操作前的旧数量 `N_old`，只对前 `N_old` 个Gaussian做额外剪枝
  - 创建长度为 `N_current` 的全False的bool Tensor `low_weight_mask`
  - 将前 `N_old` 个中 `weight_scores < weight_prune_thr` 的位置设为True
  - 用 `torch.logical_or` 把 `low_weight_mask` 并入 `prune_mask`

**注意事项：**
- `weight_scores` 和 `weight_prune_thr` 必须是可选参数，默认值分别为 `None` 和 `0.0`
- 当 `weight_scores is None` 或 `weight_prune_thr <= 0` 时，行为与原版完全一致

---

### 2.2 `train_weighted.py`

**改动位置1：** 训练循环初始化部分（权重图加载器初始化之后）

新增两个变量：
```python
gaussian_weight_accum = None   # 累积每个Gaussian的权重分数之和
gaussian_weight_count = None   # 累积每个Gaussian被看到的次数
```

---

**改动位置2：** 每次迭代中，Loss计算完成之后、`loss.backward()` 之前

当 `use_weight_map=True` 且 `weight_map is not None` 时，积累当前帧的权重分数到每个可见Gaussian：

步骤：
1. 获取 `n_gaussians = gaussians.get_xyz.shape[0]`
2. 如果 `gaussian_weight_accum` 为 None 或 shape 不匹配（clone/split导致数量变化），重新初始化为全0的Tensor
3. 复用已有的投影代码（现有densification抑制逻辑中已有），将可见Gaussian的3D坐标投影到2D像素坐标
4. 读取对应像素位置的权重值：`vis_weights = weight_map[0, py, px]`
5. 累加：
   ```python
   gaussian_weight_accum[vis_idx] += vis_weights
   gaussian_weight_count[vis_idx] += 1
   ```

---

**改动位置3：** `densify_and_prune` 调用处

调用前，计算 `weight_scores`：
```python
weight_scores = None
if gaussian_weight_accum is not None and gaussian_weight_accum.shape[0] == gaussians.get_xyz.shape[0]:
    valid = gaussian_weight_count > 0
    avg_scores = torch.ones(gaussian_weight_accum.shape[0], device="cuda")
    avg_scores[valid] = gaussian_weight_accum[valid] / gaussian_weight_count[valid]
    # 未被任何相机看到的Gaussian填1.0（不额外剪枝）
    weight_scores = avg_scores
    # 调用完后重置，下一轮重新积累
    gaussian_weight_accum = None
    gaussian_weight_count = None
```

调用 `densify_and_prune` 时新增参数：
```python
gaussians.densify_and_prune(
    opt.densify_grad_threshold, 0.005,
    scene.cameras_extent, size_threshold, radii,
    weight_scores=weight_scores,
    weight_prune_thr=getattr(opt, "weight_prune_thr", 0.2),
)
```

---

### 2.3 `arguments/__init__.py`

在 `OptimizationParams` 中新增三个参数（field形式，与已有参数风格一致）：

```python
weight_prune_thr: float = 0.2
# 说明：Gaussian平均可信度低于此阈值时，在pruning阶段被额外剔除
# 推荐范围：0.1 ~ 0.3，设为0.0则不启用

use_normal_loss: bool = False
# 说明：是否启用法向一致性约束（模块3）

lambda_normal: float = 0.01
# 说明：法向一致性loss的权重系数
# 推荐范围：0.001 ~ 0.05
```

同时在 `train_weighted.py` 的 `argparse` 部分新增对应命令行参数（参考已有 `--weight_prune_thr` 风格），并在底部的 `op_args` 赋值区域同步赋值。

---

## 模块3：权重图调制的法向一致性约束

### 涉及文件
- `utils/surface_utils.py`（**新建文件**）
- `train_weighted.py`（修改）

---

### 3.1 新建 `utils/surface_utils.py`

**函数签名：**
```python
def compute_normal_consistency_loss(
    gaussians,
    weight_scores=None,
    k_neighbors: int = 10,
    lambda_normal: float = 0.01,
    sample_size: int = 2000,
) -> torch.Tensor:
```

**实现步骤：**

**Step 1：提取每个Gaussian的法向量**
```python
xyz = gaussians.get_xyz           # [N, 3]
scales = gaussians.get_scaling    # [N, 3]，已过exp激活
rots = gaussians.get_rotation     # [N, 4]，四元数

from utils.general_utils import build_rotation
rot_mats = build_rotation(rots)   # [N, 3, 3]，列向量是局部坐标轴

min_axis_idx = scales.argmin(dim=-1)   # [N]，最小scale轴索引
N = xyz.shape[0]
normals = rot_mats[torch.arange(N), :, min_axis_idx]   # [N, 3]
normals = F.normalize(normals, dim=-1)
```

**Step 2：边界检查**
```python
if N < k_neighbors + 1:
    return torch.tensor(0.0, device=xyz.device, requires_grad=False)
```

**Step 3：随机采样，降低计算量**
```python
s = min(N, sample_size)
sample_idx = torch.randperm(N, device=xyz.device)[:s]
xyz_s = xyz[sample_idx]          # [S, 3]
normals_s = normals[sample_idx]  # [S, 3]
```

**Step 4：KNN找邻域**
```python
dist = torch.cdist(xyz_s, xyz_s)           # [S, S]
dist.fill_diagonal_(float('inf'))
_, nn_idx = dist.topk(k_neighbors, dim=-1, largest=False)   # [S, K]
```

**Step 5：计算法向一致性损失**
```python
nn_normals = normals_s[nn_idx]                          # [S, K, 3]
center_normals = normals_s.unsqueeze(1).expand_as(nn_normals)   # [S, K, 3]
cos_sim = (center_normals * nn_normals).sum(dim=-1).abs()       # [S, K]，取绝对值（法向可能反向）
normal_loss_per = (1.0 - cos_sim).mean(dim=-1)                  # [S]
```

**Step 6：权重图调制**
```python
if weight_scores is not None:
    w = weight_scores[sample_idx].clamp(0, 1).detach()  # [S]，不参与梯度
    weighted_loss = (w * normal_loss_per).mean()
else:
    weighted_loss = normal_loss_per.mean()

return lambda_normal * weighted_loss
```

**注意：**
- `weight_scores` 用 `.detach()` 避免影响梯度图
- `torch.cdist` 在 S=2000 时内存约 2000×2000×4 bytes ≈ 16MB，可接受
- 如后续N很大导致显存压力，可将 `sample_size` 调小

---

### 3.2 `train_weighted.py` 修改

**在文件顶部新增导入：**
```python
from utils.surface_utils import compute_normal_consistency_loss
```

**在Loss计算部分（`loss = ...` 之后，`loss.backward()` 之前）新增：**
```python
if use_weight_map and getattr(opt, "use_normal_loss", False):
    normal_loss = compute_normal_consistency_loss(
        gaussians,
        weight_scores=None,      # 第一版先不做调制，等基础效果验证后再接入
        k_neighbors=10,
        lambda_normal=getattr(opt, "lambda_normal", 0.01),
        sample_size=2000,
    )
    loss = loss + normal_loss
```

> **说明：** 第一版 `weight_scores=None` 表示均匀约束（对所有区域等权重）。
> 接入调制时，把本次迭代已投影的 `vis_weights` 重构为全Gaussian的分数Tensor传入即可。

---

## 实施顺序

```
第一步（优先）：模块2 - 自适应Pruning
  1. 改 arguments/__init__.py，加 weight_prune_thr / use_normal_loss / lambda_normal
  2. 改 gaussian_model.py，densify_and_prune 增加两个可选参数
  3. 改 train_weighted.py，加积累逻辑和传参
  4. 测试：python train_weighted.py -s <data> -m output/test --use_weight_map --weight_prune_thr 0.2

第二步：模块3 - 法向约束（均匀版）
  1. 新建 utils/surface_utils.py
  2. 改 train_weighted.py，引入并调用
  3. 测试：加上 --use_normal_loss --lambda_normal 0.01
  4. 观察新视角效果是否改善，喷射状是否减少

第三步（后续）：调制版法向约束
  - 将 weight_scores 接入 compute_normal_consistency_loss
  - 消融对比：均匀约束 vs 调制约束

第四步（可选）：迁移到SeaSplat底座
  - SeaSplat 的 densify_and_prune 接口与原版基本一致
  - 将本计划的所有改动平移过去即可
```

---

## 消融实验设计

| 实验组 | 命令行配置 | 目的 |
|--------|-----------|------|
| A - Baseline | 原始 `train.py` | 对照 |
| B - WeightLoss | `--use_weight_map` | 验证loss加权 |
| C - WeightDensify | B + `--weight_densify_thr 0.3` | 验证densification抑制 |
| D - WeightPrune | C + `--weight_prune_thr 0.2` | 验证自适应pruning（模块2）|
| E - NormalUniform | D + `--use_normal_loss` | 验证均匀法向约束（模块3）|
| F - NormalWeighted | E + 调制版 | 完整模型 |

---

## 关键参数建议值

| 参数 | 建议初始值 | 范围 | 说明 |
|------|-----------|------|------|
| `weight_prune_thr` | 0.2 | 0.1 ~ 0.3 | 太高剔除太多，太低没效果 |
| `lambda_normal` | 0.01 | 0.001 ~ 0.05 | 太高过度平滑，丢失细节 |
| `k_neighbors` | 10 | 8 ~ 15 | 邻域大小，影响平滑范围 |
| `sample_size` | 2000 | 1000 ~ 5000 | 每步采样Gaussian数，影响速度 |
