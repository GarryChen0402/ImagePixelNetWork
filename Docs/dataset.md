# 数据集设计

> **版本**: v0.1 | **日期**: 2026-05-24 | **所属项目**: [ImagePixelNetWork](./README.md)

## 数据需求

训练需要 **成对数据集**：自然图像（输入）→ 对应像素风格图像（目标）。

但由于获取真实成对数据成本高昂，采用以下策略构建训练数据。

## 方案一：合成数据集（主方案）

### 生成流程

```
高清图像 → 下采样 → 颜色量化 → 像素艺术风格化 → 目标图像（GT）
```

### 详细步骤

#### Step 1：高清图像采集
- 来源：Open Images、COCO、ImageNet
- 数量：10,000+ 张
- 内容：风景、建筑、物体等多样主题
- 预处理：中心裁剪到 256×256 或 512×512

#### Step 2：生成像素艺术目标

对每张高清图像应用以下处理链：

```python
def generate_pixel_art(image, target_size=64, palette_size=32):
    """
    image: PIL Image (H, W)
    target_size: 像素化后的分辨率（越小越像素化）
    palette_size: 调色板颜色数
    """
    # 1. 下采样到目标低分辨率
    small = image.resize((target_size, target_size), Image.NEAREST)

    # 2. 颜色量化（k-means 聚类到 palette_size 色）
    pixels = np.array(small).reshape(-1, 3)
    _, labels, centers = kmeans(pixels, palette_size)
    quantized = centers[labels].reshape(target_size, target_size, 3)

    # 3. 上采样回原始分辨率（NEAREST 保持块状边缘）
    result = Image.fromarray(quantized).resize(
        (image.width, image.height), Image.NEAREST
    )

    return result
```

#### Step 3：数据增强

| 增强 | 参数 | 说明 |
|------|------|------|
| 随机水平翻转 | p=0.5 | 基础增强 |
| 随机旋转 | ±10° | 小幅旋转 |
| 颜色抖动 | 亮度/对比度/饱和度 | 提高泛化能力 |
| target_size 变化 | 32, 48, 64, 96 | 让模型适应不同像素粒度 |

### 合成数据的局限性

- 合成像素艺术可能缺乏真实像素艺术的手工质感
- 颜色量化方式可能与目标风格不完全匹配
- 缺少像素艺术家特有的"抖色"（dithering）等技法

## 方案二：真实像素艺术数据集（补充方案）

收集真实的像素艺术作品，作为风格参考：

| 来源 | 说明 |
|------|------|
| Minecraft 贴图包 | 从公开的 Minecraft 资源包中提取方块贴图（16×16 或 32×32） |
| PixelJoint | 像素艺术社区，大量高质量作品 |
| Lospec | 像素艺术教程和调色板数据库 |
| OpenGameArt | 开源游戏美术资源 |

这些数据主要用于：
- 调色板的初始化和风格指导
- 对抗训练的判别器预训练
- 风格损失中 Gram 矩阵目标统计的计算

## 方案三：非成对训练（备选）

如果成对数据构建困难，可以采用 CycleGAN 风格的循环一致性训练：

- 不需要成对数据，只需要自然图像集合（Domain A）和像素艺术集合（Domain B）
- 训练两个生成器：A→B（自然→像素）和 B→A（像素→自然）
- 循环一致性损失确保 A→B→A ≈ A

## 推荐数据策略

**混合策略**：以合成数据集为主体（10K+ 张），辅以真实像素艺术数据集（1K+ 张）用于风格指导。训练初期使用合成数据让模型快速收敛，后期引入真实像素艺术数据进行微调（fine-tuning）。

## 数据目录结构

```
data/
├── raw/
│   ├── natural/           # 原始高清图像
│   └── pixel_art/         # 收集的真实像素艺术
├── processed/
│   ├── train/
│   │   ├── input/         # 训练输入（256×256 自然图像）
│   │   └── target/        # 训练目标（256×256 像素风格图像）
│   ├── val/
│   │   ├── input/
│   │   └── target/
│   └── test/
│       ├── input/
│       └── target/
└── palette/
    └── minecraft_32.npy   # 预提取的 Minecraft 32 色调色板
```
