# Changelog

## v0.2.1 (2026-05-25)

Bug 修复和训练优化。

### Bug 修复

- **train.py**: 修复 `def train(args):` 函数头缺失，导致函数体变成模块级代码
- **train.py**: 修复 `find_best_batch_size` 中 G/D backward 共享计算图，第二次 backward 报 `retain_graph` 错误
- **train.py**: 修复 `pq_opt` 在 phase 切换时 LR 从不更新（Phase 1 时仅 G 的 1/100）
- **discriminator**: 移除 `MultiScaleDiscriminator` 改用 `PatchDiscriminator` — d32 分支从未被调用，参数白占显存且收不到梯度
- **train.py**: 清理未使用的 import (`F`)、冗余 import (`find_best_batch_size` 内重复)、未使用变量 (`hi`)

### 新增功能

- **`--auto-batch`**: 自动检测 GPU 显存并调整 batch size，使用 80% headroom 防 OOM
  - 检测方法：从 32 向下二分搜索，实测 forward+backward
  - 32GB GPU (RTX 4080 SUPER) 实测：batch=32 可装入，safe=25

## v0.2 (2026-05-25)

非成对数据架构重构。从监督学习切换到 PatchGAN 对抗训练。

### 架构变更

- **训练范式**: 成对监督 → 非成对 PatchGAN 对抗训练
- **判别器**: 新增 16×16 Patch Discriminator，真样本为 Minecraft tiles
- **损失函数**: 新增 L_adv (Hinge)、L_style_patch (Gram patch)，L_content 改为对比输入-输出
- **调色板**: 32→64 色，初始化方式从随机 → k-means (Minecraft tiles)
- **训练**: 200→300 epochs，新增 Phase 1 预训练（无对抗）

### 新增文件

| 文件 | 说明 |
|------|------|
| `src/palette.py` | 可微分调色板量化 + k-means 提取 |
| `src/generator.py` | U-Net 生成器 (55M 参数) |
| `src/discriminator.py` | PatchGAN 判别器 (660K 参数) |
| `src/losses.py` | VGG 特征提取 + 6 种损失函数 |
| `src/dataset.py` | 非成对数据加载 (PhotoDataset + TileBank) |
| `src/train.py` | 三阶段对抗训练 + FP16 + checkpoint |
| `scripts/extract_palette.py` | k-means 调色板提取 |
| `scripts/compute_style_stats.py` | Gram 风格统计预计算 |
| `scripts/split_collections.py` | Collections.png 拆分 |

### 数据集

- SceneImage: 7,268 张风景照片 (320×180)
- MinecraftImage: 2,004 张 16×16 tiles (从 Collections.png 拆分)
- 调色板: k-means 64 色 (从 ~336K pixels 聚类)
- Gram 统计: relu1_1/relu2_1/relu3_1 均值

## v0.1 (2026-05-24)

初始设计文档发布。

### 文档清单

| 文档 | 说明 |
|------|------|
| `README.md` | 项目概述、技术路线、目录结构 |
| `architecture.md` | Encoder-Decoder + PaletteQuantizer 网络架构 |
| `dataset.md` | 三种数据集构建方案 |
| `training.md` | 三阶段训练策略与超参数配置 |

### 核心设计决策

- **网络结构**：编码器-解码器 + 可微分调色板量化层
- **量化方式**：温度退火软分配
- **损失函数**：内容 + 风格 + TV + 调色板
- **数据策略**：合成数据为主，真实像素艺术为辅
- **训练策略**：三阶段渐进训练
