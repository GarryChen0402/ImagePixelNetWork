# 网络结构图

> **版本**: v0.2 | **日期**: 2026-05-25 | **所属项目**: [ImagePixelNetWork](./README.md)

## 整体数据流

```
Photo (A)  →  Generator (U-Net)  →  PaletteQuantizer  →  Output
                                                           │
                                              ┌────────────▼────────────┐
                                              │  PatchGAN D (16×16)     │
                                              │  ← Minecraft tiles (真) │
                                              │  ← Output patches (假)  │
                                              └─────────────────────────┘
```

---

## 1. Generator（U-Net + PixelShuffle）

参数量 ~55M。Encoder 逐步下采样提取特征，Bottleneck 做深层变换，Decoder 通过 PixelShuffle 上采样 + 跳跃连接恢复空间细节。

```
Input Photo A (256 × 256 × 3)
  │
  ▼
Conv7×7 → 64ch                          (stem features)
  │                                         │
  │ MaxPool 2×                              │  1×1 proj → skip
  ▼                                         │
Enc Block1: 2×ResBlock → 128ch (128×128)    │
  │                                         │
  │ MaxPool 2×                              │  1×1 proj → skip
  ▼                                         │
Enc Block2: 2×ResBlock → 256ch (64×64)      │
  │                                         │
  │ MaxPool 2×                              │  1×1 proj → skip
  ▼                                         │
Enc Block3: 2×ResBlock → 512ch (32×32)      │
  │                                         │
  │ MaxPool 2×                              │  1×1 proj → skip
  ▼                                         │
Enc Block4: 2×ResBlock → 512ch (16×16)      │
  │                                         │
  ▼                                         │
Bottleneck: 6×ResBlock (512ch, 16×16)       │
  │                                         │
  │ PixelShuffle ↑×2                        │
  │ ← 1×1 proj(Enc4) → concat → fuse 1×1   │
  ▼                                         │
Dec Block3: 256ch (32×32)                   │
  │                                         │
  │ PixelShuffle ↑×2                        │
  │ ← 1×1 proj(Enc3) → concat → fuse 1×1  ─┘
  ▼
Dec Block2: 128ch (64×64)
  │
  │ PixelShuffle ↑×2
  │ ← 1×1 proj(Enc2) → concat → fuse 1×1  ──┐
  ▼                                         │
Dec Block1: 64ch (128×128)                  │
  │                                         │
  │ PixelShuffle ↑×2                        │
  │ ← 1×1 proj(Enc1) → concat → fuse 1×1  ─┘
  ▼
Dec Block0: 64ch (256×256)
  │
  │ ← 1×1 proj(stem) → concat → fuse 1×1   ──┐
  ▼                                           │
Conv7×7 → 3ch → Sigmoid                      │
  │                                           │
  ▼                                           │
PaletteQuantizer (64色调色板)                  │
  │                                           │
  ▼                                           │
Output (256 × 256 × 3)
```

### 跳跃连接细节

```
Encoder输出 ──→ 1×1 proj (对齐通道数) ──→ concat ──→ 1×1 fuse (降维) ──→ 下一层
                           ↑                               │
             Decoder上采样结果 ─────────────────────────────┘
```

- 投影卷积将不同通道数的 Encoder 输出统一对齐
- Concat 后紧跟 1×1 卷积融合，避免通道膨胀
- PixelShuffle 保证上采样锐利边缘，不使用双线性插值

---

## 2. PaletteQuantizer（可微分调色板量化）

```
Continuous RGB (256×256×3)
  │
  │  计算到 64 个可学习调色板颜色的距离
  │  dist[i] = ||rgb - palette[i]||²
  │
  ▼
训练: softmax(-dist / τ)  →  加权和  →  软量化 RGB
推理: argmin(dist)        →  查表    →  硬量化 RGB
```

| 参数 | 值 | 说明 |
|------|-----|------|
| `palette_size` | 64 | Minecraft 纹理颜色更丰富 |
| 初始化 | k-means | 从 2004 张 Minecraft tiles 聚类 |
| τ 退火 | 1.0 → 0.1 | 三阶段控制，高温软、低温硬 |

调色板本身是 `nn.Parameter`，随训练更新。

---

## 3. Patch Discriminator（纹理判别器）

参数量 ~660K。仅在 16×16 局部尺度判别纹理是否像 Minecraft tile，不关心全局结构。

```
16×16×3 patch
  │
  ▼
Conv (stride=2) → LeakyReLU(0.2)     16×16 → 8×8
  │
  ▼
Conv (stride=2) → BN → LeakyReLU     8×8 → 4×4
  │
  ▼
Conv (stride=2) → BN → LeakyReLU     4×4 → 2×2
  │
  ▼
Conv (stride=2) → BN → LeakyReLU     2×2 → 1×1
  │
  ▼
Conv1×1 → 1 (真/假)
  │
  ▼
Hinge Loss: max(0, 1-D(real)) + max(0, 1+D(fake))
R1 梯度惩罚: λ=10, 每2步应用
```

### 真/假样本来源

| 类型 | 来源 | 增强 |
|------|------|------|
| 真样本 | Minecraft tile bank (16×16 tiles) | 随机翻转、旋转、颜色抖动 |
| 假样本 | Generator 输出中随机裁剪 16×16 patches | 无 |

---

## 4. 损失函数

```
L_total =  λ_adv × L_adv
         + λ_content × L_content
         + λ_style × L_style_patch
         + λ_edge × L_edge
         + λ_tv × L_tv
         + λ_palette × L_palette
```

| 损失项 | 权重 | 类型 | VGG 层 | 说明 |
|--------|------|------|--------|------|
| `L_adv` | 0.5 | Hinge | - | PatchGAN 对抗损失，推使局部纹理像 Minecraft |
| `L_content` | 1.0 | L1 | relu4_2 | 输入照片 vs 输出的 VGG 特征差异（非成对） |
| `L_style_patch` | 3.0 | Gram | relu1_1, relu2_1, relu3_1 | 输出 patches Gram 矩阵 vs Minecraft tiles 统计均值 |
| `L_edge` | 0.5 | Laplacian | - | 保持输入照片的边缘结构 |
| `L_tv` | 1e-4 | TV | - | 总变分正则化，抑制噪声 |
| `L_palette` | 0.1 | 熵 | - | 最大化调色板使用熵，防止颜色崩塌 |

### 判别器正则化

- **Hinge Loss**（比 LSGAN 更稳定）
- **R1 梯度惩罚**: λ=10，每 2 步应用一次
- **Label Smoothing**: real label = 0.9（防止判别器过自信）

---

## 5. 训练策略（三阶段）

| 阶段 | Epochs | 策略 | τ | LR |
|------|--------|------|-----|------|
| 1 (预训练) | 1–30 | G 重建 + 量化，冻结 D | 1.0 | 1e-3 |
| 2 (对抗注入) | 31–120 | 解锁 D，逐步加大对抗权重 | 1.0 → 0.2 | 1e-4 |
| 3 (精调) | 121–200 | 全损失激活，t=0.1，余弦衰减 | 0.1 | 1e-4 → 1e-6 |

关键超参数：
- 图像尺寸: 256×256
- Batch Size: 16
- Optimizer: Adam (β₁=0.9, β₂=0.999)
- Mixed Precision: FP16
- LR Schedule: Cosine Annealing

---

## 6. 推理流程

```
Input Photo (任意分辨率)
  │
  ▼
Resize → 256×256×3
  │
  ▼
Generator Forward (硬量化, τ→0 等价 argmax)
  │
  ▼
Output (256×256×3, 限定于64色调色板)
  │
  ▼
[可选] Grid 后处理:
  分割为 16×16 grid → 每格取众数颜色 → 增强方块感
  │
  ▼
Final Pixel Art
```
