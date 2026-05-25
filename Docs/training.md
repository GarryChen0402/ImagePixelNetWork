# 训练策略

> **版本**: v0.2 | **日期**: 2026-05-25 | **所属项目**: [ImagePixelNetWork](./README.md)

## 训练配置

### 基础超参数

| 参数 | 推荐值 | 说明 |
|------|--------|------|
| Image Size | 256×256 | 训练图像分辨率 |
| Batch Size | auto (--auto-batch) | 自动检测 GPU 显存最优值，留 80% headroom |
| Optimizer (G) | Adam (β1=0.9, β2=0.999) | 标准配置 |
| Optimizer (D) | Adam (β1=0.9, β2=0.999) | 同 G |
| Mixed Precision | FP16 | 节省显存，加速训练 |
| Epochs | 300 | 总训练轮数 |

`--auto-batch` 会在训练开始前从 32 向下二分搜索最大可用 batch size。32GB GPU 实测约 25（80% headroom）。也可手动指定 `--batch-size N`。

### 损失权重动态调度

| 阶段 | L_adv | L_style_patch | L_content | L_edge | L_tv | L_palette |
|------|-------|---------------|-----------|--------|------|-----------|
| 1 (预训练) | 0.0 | 0.0 | 1.0 | 0.5 | 1e-4 | 0.1 |
| 2 (对抗注入) | 0→0.5 | 0→3.0 | 1.0 | 0.5 | 1e-4 | 0.1 |
| 3 (精细调优) | 0.5 | 3.0 | 1.0 | 0.5 | 1e-4 | 0.1 |

## 三阶段训练策略

### Phase 1：预训练（Epochs 1~50）

**目标**: Generator 学会基本的内容重建（编码→解码），生成有意义的结构化输出。

- 仅使用 L_content + L_edge + L_tv + L_palette
- **不使用对抗损失**（Discriminator 未激活）
- 调色板温度 τ = 1.0（完全软量化，梯度平滑）
- LR(G) = 1e-3（较高，快速收敛）
- 在此阶段 Generator 输出的是模糊但结构正确的重建

**为什么需要预训练？** 如果一开始就引入随机初始化的 D，D 会轻松击败 G，导致训练崩溃。先让 G 学会基本的编解码能力。

### Phase 2：对抗注入（Epochs 51~200）

**目标**: 在内容保持的基础上引入 Minecraft 纹理风格。

- 所有损失激活
- L_adv 权重从 0 线性增长到 0.5（前 80 epochs 渐进）
- L_style_patch 权重从 0 线性增长到 3.0
- 调色板温度 τ 从 1.0 退火到 0.3
- LR(G) = 1e-4, LR(D) = 4e-4
- 判别器每 2 步应用 R1 梯度惩罚
- Minecraft tile 实时增强：随机翻转、90°旋转、颜色抖动

### Phase 3：精细调优（Epochs 201~300）

**目标**: 精细化边缘，优化调色板使用，最终收敛。

- 所有损失全权重激活
- 调色板温度 τ = 0.1（接近硬量化，锐利输出）
- LR(G) = 1e-4 → 1e-6（Cosine Annealing）
- LR(D) = 1e-4 → 1e-6

### 温度退火曲线

```
Epoch   1: τ = 1.00  (Phase 1 — 完全软量化)
Epoch  50: τ = 1.00
Epoch  80: τ = 0.60  (Phase 2 — 退火中)
Epoch 140: τ = 0.40
Epoch 200: τ = 0.30
Epoch 250: τ = 0.15  (Phase 3 — 接近硬量化)
Epoch 300: τ = 0.10
```

### 判别器训练细节

- D 更新频率: 每次 G 更新后更新 1 次 D（1:1 比例）
- 真样本数 = 假样本数 = B × 4（每个 batch 采样 4 个随机 patches/图片）
- R1 惩罚权重: 10.0
- Label smoothing: real=0.9, fake=0.0

## 监控指标

| 指标 | 说明 | 期望趋势 |
|------|------|---------|
| G_total | Generator 总损失 | ↓ 下降 |
| L_adv (G) | 对抗损失 | 在 0 附近振荡（健康 GAN） |
| L_content | 内容保真度 | ↓ 下降后稳定 |
| L_style_patch | 纹理匹配度 | ↓ 下降后稳定 |
| L_d_adv | 判别器损失 | 在 1~2 之间振荡 |
| Palette Usage | 调色板使用熵 | 接近 log(64) ≈ 4.16 |
| τ | 温度参数 | 按计划退火 |

### 颜色崩塌（Color Collapse）预警

如果 Palette Usage 持续下降：
1. 增大 L_palette 权重（0.1 → 0.5）
2. 降低学习率
3. 检查调色板初始化是否合理

### 训练崩溃预警

如果 L_adv(G) 持续上升而 L_d_adv 趋近于 0（D 过强）：
1. 降低 LR(D) 或增大 LR(G)
2. 减少 D 更新频率（2:1 G:D 比例）
3. 检查 R1 惩罚是否生效

## 推理配置

```
- 调色板量化：硬分配（argmax）
- BatchNorm → eval 模式
- 可选 Grid 后处理：16×16 grid 颜色众数
- 输入尺寸：256×256（任意比例 resize）
```

## 硬件需求

| 组件 | 最低要求 | 推荐配置 |
|------|---------|---------|
| GPU | 8GB VRAM | 16GB+ VRAM (RTX 4080+) |
| RAM | 16GB | 32GB |
| Disk | 5GB | 10GB (含数据集和检查点) |

## 预估训练时间

| GPU | 300 Epochs 预估时间 |
|-----|-------------------|
| RTX 4090 | ~8-10 小时 |
| RTX 4080 | ~12-15 小时 |
| RTX 3080 | ~20-25 小时 |
| RTX 3060 | ~30-40 小时 |

## 训练命令

```bash
python -m src.train \
  --photo-dir Datasets/SceneImage/landscape_dataset \
  --tile-dir Datasets/MinecraftImage/tiles \
  --palette Datasets/palette/minecraft_64.npy \
  --gram-stats Datasets/style/minecraft_gram_stats.pt \
  --output outputs/run01 \
  --auto-batch \
  --epochs 300 \
  --fp16
```

## 恢复训练

```python
ckpt = torch.load("outputs/run01/checkpoints/ckpt_epoch0100.pt")
g.load_state_dict(ckpt["g"])
d.load_state_dict(ckpt["d"])
pq.load_state_dict(ckpt["pq"])
g_opt.load_state_dict(ckpt["g_opt"])
d_opt.load_state_dict(ckpt["d_opt"])
start_epoch = ckpt["epoch"]
```
