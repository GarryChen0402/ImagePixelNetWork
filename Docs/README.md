# ImagePixelNetWork

> **版本**: v0.2 | **日期**: 2026-05-25 | **状态**: 实现阶段

将输入图片转换为像素风格（Minecraft 贴图风格）的深度学习网络。

## 项目目标

输入任意分辨率的自然图像，输出具有像素艺术风格的图像。核心特征：

- **块状纹理**: 输出图像呈现 Minecraft 贴图的方块化风格
- **有限调色板**: 颜色量化到 64 色离散调色板
- **锐利边缘**: 无抗锯齿，保留像素艺术的硬边缘
- **内容保真**: 在风格化的同时保持原始图像的语义结构

## 技术路线

v0.2 采用 **PatchGAN 非成对对抗训练** 框架：

```
Photo (256×256) → Generator (U-Net) → PaletteQuantizer → Pixel Art 输出
                                          ↑
                                   PatchGAN D (16×16)
                                   ✓ 真 = Minecraft tiles
                                   ✗ 假 = 输出随机 patches
```

### 为什么不用成对监督学习？

获取同一场景的"自然照片 vs 像素艺术"成对数据成本极高。我们拥有的数据是：7,268 张自然风景照 + 2,004 张 16×16 Minecraft 纹理块。PatchGAN 框架让判别器在**局部纹理块**级别学习"Minecraft 风格"，而非整体图像。

### 关键技术点

1. **PatchGAN 纹理判别器**: 16×16 感受野，直接在 Minecraft tiles 上训练，使 Generator 输出的每个局部 patch 都具有 Minecraft 质感
2. **可微分调色板量化**: 通过软分配（温度退火）映射连续颜色到 64 色调色板，端到端可训练
3. **内容保持**: VGG 感知损失对比输入-输出（非成对），配合 U-Net 跳跃连接保持场景结构

## 文档索引

| 文档 | 说明 |
|------|------|
| [CHANGELOG.md](./CHANGELOG.md) | 版本变更记录 |
| [architecture.md](./architecture.md) | 网络架构设计 (PatchGAN) |
| [dataset.md](./dataset.md) | 非成对数据集设计 |
| [training.md](./training.md) | 三阶段对抗训练策略 |

## 目录结构

```
ImagePixelNetWork/
├── Docs/                    # 设计文档
├── src/                     # 源代码
│   ├── palette.py           # 可微分调色板量化
│   ├── generator.py         # U-Net 生成器
│   ├── discriminator.py     # PatchGAN 判别器
│   ├── losses.py            # 损失函数
│   ├── dataset.py           # 非成对数据加载
│   └── train.py             # 训练入口
├── scripts/                 # 辅助脚本
│   ├── split_collections.py     # 拆分精灵表
│   ├── extract_palette.py       # 提取 k-means 调色板
│   └── compute_style_stats.py   # 计算 Gram 风格统计
├── Datasets/                # 数据集
│   ├── SceneImage/          # 风景照片 (7,268)
│   ├── MinecraftImage/      # Minecraft 纹理 (2,004 tiles)
│   ├── palette/             # 预提取调色板
│   └── style/               # 预计算 Gram 统计
├── checkpoints/             # 模型权重
└── outputs/                 # 训练输出 + 样本
```

## 快速开始

```bash
# 1. 提取调色板
python scripts/extract_palette.py

# 2. 计算风格统计
python scripts/compute_style_stats.py

# 3. 开始训练
python -m src.train --fp16 --batch-size 8 --epochs 300
```

## 环境依赖

- Python 3.10+
- PyTorch 2.0+
- torchvision
- scikit-learn
- Pillow
- NumPy
- tqdm
