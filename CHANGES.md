# 改动说明文档

## 项目背景

本项目在原始 3D Gaussian Splatting（3DGS）代码基础上，针对**水下场景重建**问题进行改进。

水下图像的核心挑战是：
- 远处场景因光线散射、吸收，图像模糊、色偏严重，信息可信度低
- 近处场景相对清晰，信息可信度高
- 原始 3DGS 训练时对所有像素**均等加权**，导致低可信度区域（远景/散射区）产生大量伪 Gaussian（浮影/噪点）

**改进思路**：在训练过程中引入逐像素的**可信度权重图**，让高可信度区域的像素贡献更多梯度，低可信度区域的区域减少对 Gaussian 生长的影响。

---

## 文件改动清单

| 文件 | 类型 | 说明 |
|------|------|------|
| `train.py` | **未改动** | 原始 3DGS 训练脚本，完整保留，用于跑基线对比 |
| `train_original.py` | 备份 | `train.py` 的备份副本，可删除 |
| `train_weighted.py` | **新增** | 改进版训练脚本，加入权重图逻辑 |
| `utils/weight_map_utils.py` | **新增** | 权重图计算和加载工具库 |

**原则：原始 `train.py` 和其余所有文件完全未改动。** 两套脚本可以在同一个代码库中并行运行，方便对比实验。

---

## 原始 3DGS 训练流程（`train.py`）

### 核心循环逻辑

```
for each iteration:
    1. 随机采样一个训练相机视角 viewpoint_cam
    2. 渲染：image = render(viewpoint_cam, gaussians)
    3. 计算 Loss（均等加权）：
         Ll1   = mean(|image - gt_image|)            ← 所有像素权重相同
         ssim  = SSIM(image, gt_image)
         loss  = (1 - λ) * Ll1 + λ * (1 - ssim)
    4. 反向传播：loss.backward()
    5. Densification（Gaussian 增殖/剪枝）：
         - 所有可见 Gaussian 都参与梯度统计
         - 梯度大的 Gaussian → 分裂（clone/split）
         - 透明度低的 Gaussian → 剪枝
    6. 优化器更新参数
```

### 关键问题

在水下场景中，**远景模糊区域的像素**和**近景清晰区域的像素**对 Loss 的贡献完全相同。  
这导致：
- 模糊区域的渲染误差较大，但这些误差来自"图像本身质量差"而非"重建失败"
- 3DGS 为了拟合这些低质量像素，在散射区域生成大量伪 Gaussian（浮影）
- 最终重建结果在近景清晰目标附近出现噪点和伪影

---

## 改进后的训练流程（`train_weighted.py`）

### 新增模块：可信度权重图

权重图 `W(x)` 由两部分融合得到：

```
W(x) = W_sharp(x)^α × t(x)^β

W_sharp(x)：清晰度权重
  → 对图像做拉普拉斯算子，计算局部梯度能量
  → 越清晰（边缘越丰富）→ 值越接近 1
  → 越模糊（散射区、远景）→ 值越接近 0

t(x)：传输图（距离权重）
  → 基于水下暗通道先验（UDCP）估计
  → 像素越靠近相机（衰减越少）→ t(x) 越大 → 值越接近 1
  → 像素越远（衰减越严重）→ t(x) 越小 → 值越接近 0

α, β：超参数，控制两个分量的相对影响强度
```

权重图为每张训练图像生成一张与之同分辨率的浮点图，值域 `[0, 1]`，无需人工标注，完全由物理公式自动计算。

### 改进后的核心循环逻辑

```
for each iteration:
    1. 随机采样一个训练相机视角 viewpoint_cam
    2. 渲染：image = render(viewpoint_cam, gaussians)
    3. 加载/计算权重图：W = WeightMapLoader.get(viewpoint_cam)   ← 新增
    4. 计算 Loss（逐像素加权）：
         Ll1   = mean(W × |image - gt_image|)         ← 高可信区贡献大梯度
         ssim  = SSIM(image, gt_image)                ← SSIM 保持不变
         loss  = (1 - λ) * Ll1 + λ * (1 - ssim)
    5. 反向传播：loss.backward()
    6. Densification（带权重抑制）：                   ← 新增
         - 对每个可见 Gaussian，采样其 2D 投影位置的权重值
         - 权重 > threshold 的 Gaussian 才参与梯度统计（可参与分裂）
         - 权重 ≤ threshold 的 Gaussian 不参与分裂统计（抑制伪 Gaussian 增殖）
    7. 优化器更新参数
```

### 两处改动的作用

| 改动位置 | 作用 | 效果 |
|----------|------|------|
| **Loss 加权** | 高可信区贡献大梯度，低可信区贡献小梯度 | 减少远景模糊区域对优化的干扰，Loss 更专注于近景高质量区域 |
| **Densification 抑制** | 低权重位置的 Gaussian 不触发分裂 | 抑制散射区/远景伪 Gaussian 的产生，减少浮影数量 |

---

## 新增命令行参数（仅 `train_weighted.py`）

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--use_weight_map` | False（flag） | 启用权重图（不加此参数则等价于原始 3DGS） |
| `--weight_map_mode` | `online` | `online`=实时计算，`precomputed`=从目录加载预计算图 |
| `--weight_map_alpha` | `1.0` | 清晰度权重指数 α |
| `--weight_map_beta` | `1.0` | 传输图权重指数 β |
| `--weight_map_dir` | `weight_maps` | 预计算权重图目录名（位于数据集根目录下） |
| `--weight_densify_thr` | `0.3` | Densification 抑制阈值，低于此值的区域不分裂 |

> 注意：`train_weighted.py` **不传 `--use_weight_map`** 时，行为与原始 `train.py` **完全一致**，可作为完整性验证。

---

## 对比实验运行方式

```bash
# 基线：原始 3DGS（完全不变）
python train.py \
  -s /path/to/underwater/dataset \
  -m output/baseline \
  --test_iterations 7000 30000 \
  --save_iterations 7000 30000

# 改进版：权重图引导（在线计算模式）
python train_weighted.py \
  -s /path/to/underwater/dataset \
  -m output/weighted \
  --use_weight_map \
  --weight_map_alpha 1.0 \
  --weight_map_beta 1.0 \
  --weight_densify_thr 0.3 \
  --test_iterations 7000 30000 \
  --save_iterations 7000 30000
```

---

## `utils/weight_map_utils.py` 模块说明

```
WeightMapLoader                   主接口，管理权重图的获取和缓存
  └── get(viewpoint_cam)          根据相机对象返回权重图 Tensor (1, H, W)

compute_weight_map(img_bgr)       融合入口：W = sharp^α × trans^β
  ├── compute_sharpness_weight()  拉普拉斯方差 → 清晰度图
  └── compute_transmission_weight() UDCP 暗通道先验 → 传输图
```

两种加载模式：
- **online**：每张图像在第一次使用时实时计算，结果缓存在内存中，无需预处理步骤
- **precomputed**：从 `<source_path>/weight_maps/` 目录加载预计算好的 `.npy` 或 `.png` 文件，速度更快，适合正式实验

---

## 依赖

改动只新增了一个依赖：

```bash
pip install opencv-python
```

其余依赖与原始 3DGS 完全相同。

---

## 后续计划

- [ ] 在 SeaThru-NeRF 数据集上跑基线 vs 改进版对比
- [ ] 在 Submerged3D 数据集（RUSplatting 提供）上测试
- [ ] 调参：α/β/densify_thr 的消融实验
- [ ] 考虑迁移到 WaterSplatting 代码库（颜色校正 + 权重图双重改进）
