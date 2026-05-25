# 网络架构设计

> **版本**: v0.2 | **日期**: 2026-05-25 | **所属项目**: [ImagePixelNetWork](./README.md)

## 整体架构

v0.2 切换为 **非成对对抗训练框架**（因无法获得成对数据）。

```
┌─────────────┐     ┌──────────┐     ┌──────────┐     ┌──────────────────┐
│  Photo (A)  │ ──→ │ Generator│ ──→ │ Decoder  │ ──→ │ PaletteQuantizer │ ──→ 输出
└─────────────┘     │ (U-Net)  │     │(PixelShuffle)│  │   (64色调色板)    │
                    └──────────┘     └──────────┘     └──────────────────┘
                                                            │
                                                    ┌───────▼────────┐
                                                    │ PatchGAN D     │ ← Minecraft 16×16 tiles (真)
                                                    │ (16×16 判别器)  │ ← 输出随机 16×16 patches (假)
                                                    └────────────────┘
```

**核心思路**: 不对输出图像整体做真假判别，而是判别 **16×16 局部纹理块** 是否像 Minecraft 贴图。Generator 的 U-Net 结构 + 跳跃连接保证空间结构，PatchGAN 保证局部纹理风格。

## 1. Generator（生成器）

U-Net 结构，Encoder → Bottleneck → Decoder，PixelShuffle 上采样 + 跳跃连接。

```
Input (256×256×3)
  → Conv7×7 (64ch)
  → Enc Block1: 2×ResBlock + MaxPool2× → 128ch, 128×128
  → Enc Block2: 2×ResBlock + MaxPool2× → 256ch, 64×64
  → Enc Block3: 2×ResBlock + MaxPool2× → 512ch, 32×32
  → Enc Block4: 2×ResBlock + MaxPool2× → 512ch, 16×16
  → Bottleneck: 6×ResBlock (512ch, 16×16)
  → Dec Block3: PixelShuffle↑(256ch) + concat(Enc3 proj) → fuse 1×1 → 256ch, 32×32
  → Dec Block2: PixelShuffle↑(128ch) + concat(Enc2 proj) → fuse 1×1 → 128ch, 64×64
  → Dec Block1: PixelShuffle↑(64ch) + concat(Enc1 proj) → fuse 1×1 → 64ch, 128×128
  → Dec Block0: PixelShuffle↑(64ch) + concat(Stem proj) → fuse 1×1 → 64ch, 256×256
  → Conv7×7 → 3ch → Sigmoid
  → PaletteQuantizer(palette_size=64, τ)
  → Output (256×256×3)
```

**设计要点**：
- 跳跃连接使用 1×1 投影卷积对齐通道数，然后 concat + 1×1 fusion 降维
- PixelShuffle 上采样保证锐利边缘，不使用双线性插值
- 总参数量：~55M

## 2. Patch Discriminator（纹理判别器）

仅在 16×16 尺度上做真假判别，不关心全局结构。

- 输入：16×16×3 的 patches
- 结构：4 层 Conv-BN-LeakyReLU (stride=2)，无池化
- 感受野：16×16
- 参数量：~660K
- **真样本**: Minecraft tile bank 中随机采样（含翻转、旋转、颜色抖动增强）
- **假样本**: Generator 输出中随机裁剪 16×16 patches

## 3. PaletteQuantizer（可微分调色板量化层）

与 v0.1 设计保持一致，参数调整：

| 参数 | v0.1 | v0.2 | 说明 |
|------|------|------|------|
| `palette_size` | 32 | 64 | Minecraft 纹理颜色更丰富 |
| 初始化 | 随机 | k-means | 从 2004 张 Minecraft tiles 聚类初始化 |
| 温度退火 | 1.0→0.1 | 1.0→0.1 | 策略不变，三阶段控制 |

调色板本身是可学习参数，训练时通过软分配（softmax(-dist/τ)）保持可微，推理时使用硬分配（argmax）。

## 4. 损失函数（重新设计）

```
L_total = λ_adv × L_adv + λ_content × L_content + λ_style_patch × L_style_patch
        + λ_edge × L_edge + λ_tv × L_tv + λ_palette × L_palette
```

| 损失项 | 权重 | 说明 |
|--------|------|------|
| L_adv | 0.5 | PatchGAN Hinge 对抗损失，推使局部纹理像 Minecraft |
| L_content | 1.0 | VGG L1 (relu4_2)，对比 **输入照片 vs 输出**（非成对） |
| L_style_patch | 3.0 | 输出随机 patches 的 Gram 矩阵 vs Minecraft tiles 统计均值 |
| L_edge | 0.5 | Laplacian 边缘保持，继承输入照片的边缘结构 |
| L_tv | 1e-4 | 总变分正则化 |
| L_palette | 0.1 | 调色板使用熵最大化（防止颜色崩塌） |

### VGG 特征层

- L_content: `relu4_2`
- L_style_patch: `relu1_1, relu2_1, relu3_1`

### 判别器正则化

- Hinge Loss（比 LSGAN 更稳定）
- R1 梯度惩罚（λ=10, 每 2 步应用）
- Label smoothing: real=0.9

## 5. 推理后处理

与 v0.1 相同，可选的 Grid 后处理：
1. 将输出分割为 16×16 的 grid
2. 每个 grid 取众数颜色
3. 增强方块化视觉效果
