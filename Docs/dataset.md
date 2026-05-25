# 数据集设计

> **版本**: v0.2 | **日期**: 2026-05-25 | **所属项目**: [ImagePixelNetWork](./README.md)

## 数据策略

v0.2 采用 **非成对训练**（unpaired training），不需要自然图像→像素艺术的成对数据。

## 数据集构成

### Domain A: 风景照片（SceneImage）

| 属性 | 值 |
|------|-----|
| 路径 | `Datasets/SceneImage/landscape_dataset/` |
| 数量 | 7,268 张 |
| 格式 | JPG RGB |
| 分辨率 | 320×180（训练时 resize 至 256×256） |
| 内容 | 自然风景（天空、山脉、水面、树木等） |

**预处理**：
- 始终保持 RGB 3 通道
- 归一化到 [0, 1]
- Resize 到 256×256（LANCZOS 插值）

### Domain B: Minecraft 像素纹理（MinecraftImage）

| 属性 | 值 |
|------|-----|
| 路径 | `Datasets/MinecraftImage/tiles/` |
| 数量 | 2,004 张（从 Collections.png 拆分，跳过空白块） |
| 格式 | PNG RGBA（alpha 合成白底后取 RGB） |
| 分辨率 | 16×16 |
| 内容 | Minecraft 方块贴图纹理 |

**来源**: `Collections.png`（1024×1024 精灵表）→ `scripts/split_collections.py` 拆分为 64×64=4096 个 16×16 图块，跳过透明块后保留 2004 张。

**增强**（训练时实时）：
- 随机水平/垂直翻转 (p=0.5)
- 随机 90° 旋转 (p=0.25 each)
- 轻微颜色抖动 (brightness ±0.02)

## 辅助数据

### 调色板

| 属性 | 值 |
|------|-----|
| 路径 | `Datasets/palette/minecraft_64.npy` |
| 生成 | `scripts/extract_palette.py` |
| 方法 | k-means 聚类 (64 色)，从 2,004 张 tiles 的 ~336K 像素中提取 |

### Gram 风格统计

| 属性 | 值 |
|------|-----|
| 路径 | `Datasets/style/minecraft_gram_stats.pt` |
| 生成 | `scripts/compute_style_stats.py` |
| 内容 | VGG relu1_1, relu2_1, relu3_1 层的平均 Gram 矩阵 |
| 用途 | L_style_patch 损失的目标统计 |

## 为什么不使用成对数据？

获取成对数据（同一场景的自然照片 + 像素艺术版本）成本极高。已有的两批数据虽然不成对，但通过 PatchGAN 框架可以很好地利用：

- **风景照片**提供场景结构和语义信息（通过 L_content 保持）
- **Minecraft 纹理块**提供局部纹理风格（通过 PatchGAN D 和 L_style_patch 注入）
- Generator 的 U-Net + 跳跃连接天然保持空间结构

## 数据目录结构

```
Datasets/
├── SceneImage/
│   └── landscape_dataset/   # 7,268 张风景 JPG (320×180)
├── MinecraftImage/
│   ├── Collections.png      # 原始精灵表 (1024×1024)
│   └── tiles/               # 拆分后的 16×16 tiles (2,004 PNG)
├── palette/
│   └── minecraft_64.npy     # k-means 64 色调色板
└── style/
    └── minecraft_gram_stats.pt  # Gram 矩阵统计
```
