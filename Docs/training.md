# 训练策略

> **版本**: v0.1 | **日期**: 2026-05-24 | **所属项目**: [ImagePixelNetWork](./README.md)

## 训练配置

### 基础超参数

| 参数 | 推荐值 | 说明 |
|------|--------|------|
| Image Size | 256×256 | 训练图像分辨率 |
| Batch Size | 16 | 根据 GPU 显存调整 |
| Optimizer | Adam | β1=0.9, β2=0.999 |
| Learning Rate | 1e-4 | 初始学习率 |
| LR Schedule | Cosine Annealing | 周期 100 epochs |
| Epochs | 200 | 总训练轮数 |
| Mixed Precision | FP16 | 节省显存，加速训练 |

### 损失权重配置

| 损失 | 权重 | 说明 |
|------|------|------|
| Content Loss | 1.0 | 内容保真 |
| Style Loss | 10.0 | 风格迁移强度 |
| TV Loss | 1e-4 | 平滑正则 |
| Palette Loss | 0.1 | 调色板均衡 |
| Edge Loss | 0.5 | 锐利边缘 |

## 三阶段训练策略

### Phase 1：预训练（Epochs 1~30）

**目标**：让模型学会基本的内容重建和颜色量化。

- 仅使用 L1 重建损失 + 调色板损失
- 调色板温度 τ 设为 1.0（软量化）
- 不使用风格损失（此时模型需要先学会"看清"内容）
- 学习率：1e-3（较高，快速收敛）

### Phase 2：风格注入（Epochs 31~120）

**目标**：在内容重建的基础上引入像素艺术风格。

- 逐步引入风格损失（L_style 权重从 0 线性增长到 10）
- 逐步降低调色板温度 τ（从 1.0 退火到 0.2）
- 引入边缘损失
- 学习率：1e-4

### Phase 3：精细调优（Epochs 121~200）

**目标**：精细化像素边缘，优化调色板使用。

- 全局损失均激活
- 调色板温度 τ 降至 0.1（接近硬量化）
- 使用学习率衰减（Cosine Annealing 到 1e-6）
- 如果使用了真实像素艺术数据集，在此阶段引入微调

### 温度退火曲线

```
Epoch  1: τ = 1.00  (完全软量化，梯度平滑)
Epoch 30: τ = 0.80
Epoch 60: τ = 0.50
Epoch 90: τ = 0.30
Epoch120: τ = 0.20
Epoch150: τ = 0.15
Epoch200: τ = 0.10  (接近硬量化，锐利输出)
```

## 监控指标

### 需要监控的关键指标

| 指标 | 说明 | 期望趋势 |
|------|------|---------|
| Total Loss | 综合损失 | ↓ 下降 |
| Content Loss | 内容保真度 | ↓ 下降后稳定 |
| Style Loss | 风格匹配度 | ↓ 下降后稳定 |
| Palette Usage | 调色板颜色使用熵 | 接近 log(palette_size)，避免颜色崩塌 |
| PSNR | 峰值信噪比（与合成 GT 对比） | ↑ 上升 |
| SSIM | 结构相似性 | ↑ 上升，>0.7 |
| FID（可选） | 与真实像素艺术分布的距离 | ↓ 下降 |

### 颜色崩塌（Color Collapse）预警

如果 Palette Usage 持续下降，说明调色板中大部分颜色未被使用（网络趋于使用少数颜色），需要：
1. 增大 Palette Loss 权重
2. 降低学习率
3. 检查调色板初始化

## 推理配置

```
模式切换：
  - 调色板量化：硬分配（argmax，非 softmax）
  - BatchNorm → 运行统计（非训练统计）
  - 可选 Grid 后处理：16×16 grid 颜色众数
```

## 硬件需求

| 组件 | 最低要求 | 推荐配置 |
|------|---------|---------|
| GPU | 8GB VRAM | 16GB+ VRAM (RTX 4080+) |
| RAM | 16GB | 32GB |
| Disk | 20GB | 50GB (含数据集) |

## 预估训练时间

| GPU | 200 Epochs 预估时间 |
|-----|-------------------|
| RTX 4090 | ~6-8 小时 |
| RTX 4080 | ~10-12 小时 |
| RTX 3080 | ~15-20 小时 |
| RTX 3060 | ~24-30 小时 |

## 训练流程伪代码

```python
model = PixelArtNetwork(palette_size=32)
optimizer = Adam(model.parameters(), lr=1e-3)
scheduler = CosineAnnealingLR(optimizer, T_max=200)

for epoch in range(1, 201):
    # 计算当前阶段的温度
    tau = get_temperature(epoch)

    # 计算当前阶段的损失权重
    weight_style = get_style_weight(epoch)

    for batch in dataloader:
        input_img, target_img = batch
        output = model(input_img, temperature=tau)

        loss_content = vgg_content_loss(output, target_img)
        loss_style   = weight_style * gram_style_loss(output, target_img)
        loss_tv      = tv_loss(output)
        loss_palette = palette_entropy_loss(model.palette)

        total_loss = (
            1.0  * loss_content +
            10.0 * loss_style +
            1e-4 * loss_tv +
            0.1  * loss_palette
        )

        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

    scheduler.step()

    # 每 5 个 epoch 保存检查点并记录指标
    if epoch % 5 == 0:
        save_checkpoint(model, optimizer, epoch)
        log_metrics(epoch, losses, images)
```
