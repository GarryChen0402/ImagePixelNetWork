# ImagePixelNetWork

> **版本**: v0.1 | **日期**: 2026-05-24 | **状态**: 设计阶段

将输入图片转换为像素风格（Minecraft 贴图风格）的深度学习网络。

## 项目目标

输入任意分辨率的自然图像，输出具有像素艺术风格的图像。核心特征：

- **块状结构**：输出图像呈现类似 Minecraft 贴图的方块化纹理
- **有限调色板**：颜色被量化到预定义的离散调色板中（16~64 色）
- **锐利边缘**：无抗锯齿，保留像素艺术的硬边缘特征
- **内容保真**：在风格化的同时保持原始图像的语义结构

## 技术路线

采用 **编码器-解码器 + 可微分调色板量化** 的端到端方案：

```
输入图像 → Encoder → Bottleneck → Decoder → Palette Quantization → 像素风格输出
```

### 为什么不用纯后处理（下采样+最近邻上采样）？

简单的下采样+最近邻插值虽然能产生块状效果，但会丢失大量细节且无法学习风格特征。
本项目的网络通过学习端到端的映射，能够在保留语义信息的同时生成更具表现力的像素风格。

### 关键技术点

1. **可微分颜色量化（Differentiable Color Quantization）**：通过软分配（Soft Assignment）机制，让网络输出可微分地映射到离散调色板，使端到端训练成为可能
2. **感知损失（Perceptual Loss）**：使用 VGG 特征空间的距离作为损失，保持内容结构
3. **风格损失（Style Loss）**：基于 Gram 矩阵匹配像素艺术的纹理统计特征
4. **总变分损失（Total Variation Loss）**：抑制噪声，保持区域平滑

## 文档索引

| 文档 | 说明 |
|------|------|
| [CHANGELOG.md](./CHANGELOG.md) | 版本变更记录 |
| [architecture.md](./architecture.md) | 网络架构设计 |
| [dataset.md](./dataset.md) | 数据集设计与构建方案 |
| [training.md](./training.md) | 训练策略与超参数配置 |

## 目录结构

```
ImagePixelNetWork/
├── Docs/                    # 设计文档
│   ├── README.md            # 项目概述（本文件）
│   ├── CHANGELOG.md         # 版本变更记录
│   ├── architecture.md      # 网络架构设计
│   ├── dataset.md           # 数据集设计
│   └── training.md          # 训练策略
├── src/                     # 源代码（待实现）
├── data/                    # 数据集（待准备）
├── checkpoints/             # 模型权重
└── outputs/                 # 推理输出
```

## 环境依赖

- Python 3.10+
- PyTorch 2.0+
- torchvision
- OpenCV / PIL
- NumPy
