# ImagePixelNetWork

将自然风景照片转换为 Minecraft 像素艺术风格的深度学习网络。

基于 **PatchGAN 非成对对抗训练** 框架：U-Net 生成器 + 16×16 纹理判别器 + 可微分 64 色调色板量化。

---

## 目录

- [环境配置](#环境配置)
- [数据集准备](#数据集准备)
- [预计算调色板与风格统计](#预计算调色板与风格统计)
- [本地训练](#本地训练)
- [远程 GPU 训练](#远程-gpu-训练)
- [监控与评估](#监控与评估)
- [推理导出](#推理导出)
- [常见问题](#常见问题)

---

## 环境配置

### 本地环境

```bash
# Python 3.10+
python --version

# 安装 PyTorch (根据你的 CUDA 版本选择)
# CUDA 12.1:
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
# CUDA 11.8:
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
# CPU only:
pip install torch torchvision

# 安装其他依赖
pip install scikit-learn pillow numpy tqdm tensorboard
```

### 远程 GPU 服务器（一键配置）

```bash
# 1. 编辑连接信息
vim scripts/remote_config.sh
# REMOTE_HOST="your-gpu-server.com"
# REMOTE_USER="your-username"

# 2. 一键配置远端 Python 环境 (CUDA + PyTorch + 依赖)
./scripts/remote_setup.sh
```

`remote_setup.sh` 会自动检测 CUDA 版本、创建 venv、安装 PyTorch 和全部依赖。

---

## 数据集准备

### 数据需求

训练需要两类**非成对**数据：

| 数据集 | 内容 | 数量 | 尺寸 |
|--------|------|------|------|
| SceneImage | 自然风景照片 | 7,268 | 320×180 (训练时 resize 到 256×256) |
| MinecraftImage | Minecraft 方块纹理 | 2,004 | 16×16 |

### 准备 Minecraft 纹理

```bash
# 将精灵表拆分为 16×16 tiles
python scripts/split_collections.py \
  Datasets/MinecraftImage/Collections.png \
  -o Datasets/MinecraftImage/tiles \
  --skip-transparent
```

跳过透明块后约保留 2,004 张有效纹理。

### 准备风景照片

将你的风景图片放入 `Datasets/SceneImage/landscape_dataset/`，支持 JPG/PNG/WebP 格式。

### 上传到远程服务器

```bash
# 首次上传所有数据集（tar+ssh 高效传输）
./scripts/upload_datasets.sh
```

---

## 预计算调色板与风格统计

训练前需要预先计算两个辅助数据：

### 1. 调色板 (k-means)

从 Minecraft tiles 中提取 64 种代表色：

```bash
python scripts/extract_palette.py \
  --tile-dir Datasets/MinecraftImage/tiles \
  --palette-size 64 \
  -o Datasets/palette/minecraft_64.npy
```

输出：`Datasets/palette/minecraft_64.npy` (64×3 float32, RGB in [0,1])

### 2. Gram 风格统计

计算 Minecraft tiles 在 VGG 特征空间的 Gram 矩阵均值：

```bash
python scripts/compute_style_stats.py \
  --tile-dir Datasets/MinecraftImage/tiles \
  --device cuda \
  -o Datasets/style/minecraft_gram_stats.pt
```

输出：`Datasets/style/minecraft_gram_stats.pt` (包含 relu1_1/relu2_1/relu3_1 三层的 Gram 均值)

> **在远端 GPU 上运行**：部署代码后，SSH 到远端执行上述两条命令，或将其加入 `remote_setup.sh` 末尾。

---

## 本地训练

### 确认一切就绪

```bash
# 检查文件结构
ls Datasets/SceneImage/landscape_dataset/ | wc -l    # 应 > 0
ls Datasets/MinecraftImage/tiles/ | wc -l             # 应 ≈ 2004
ls Datasets/palette/minecraft_64.npy                  # 应存在
ls Datasets/style/minecraft_gram_stats.pt             # 应存在
```

### 启动训练

```bash
# 基础训练
python -m src.train \
  --batch-size 8 \
  --epochs 300 \
  --fp16 \
  --output outputs/run01

# 大显存 GPU 可增大 batch size
python -m src.train \
  --batch-size 16 \
  --epochs 300 \
  --fp16 \
  --output outputs/run02
```

### 训练参数说明

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--batch-size` | 8 | 批大小，16GB 显存推荐 8~12 |
| `--epochs` | 300 | 总训练轮数（三阶段：50 + 150 + 100） |
| `--lr-g` | 1e-4 | 生成器学习率（Phase 1 为 10×） |
| `--lr-d` | 4e-4 | 判别器学习率 |
| `--image-size` | 256 | 训练分辨率 |
| `--palette-size` | 64 | 调色板颜色数 |
| `--fp16` | on | 混合精度训练（省显存，加速） |
| `--save-every` | 10 | 每 N 个 epoch 保存 checkpoint |
| `--sample-every` | 5 | 每 N 个 epoch 保存样品图片 |
| `--output` | outputs/train | 输出目录 |

### 恢复训练

```python
import torch
from src.generator import Generator
from src.discriminator import MultiScaleDiscriminator
from src.palette import PaletteQuantizer

ckpt = torch.load("outputs/run01/checkpoints/ckpt_epoch0100.pt")

g = Generator().cuda()
d = MultiScaleDiscriminator().cuda()
pq = PaletteQuantizer(palette_size=64).cuda()

g.load_state_dict(ckpt["g"])
d.load_state_dict(ckpt["d"])
pq.load_state_dict(ckpt["pq"])
start_epoch = ckpt["epoch"]
```

---

## 远程 GPU 训练

完整工作流程：

```bash
# 1. 编辑配置（填写你的服务器地址和用户名）
vim scripts/remote_config.sh

# 2. 首次：配置远端环境
./scripts/remote_setup.sh

# 3. 首次：上传数据集
./scripts/upload_datasets.sh

# 4. 部署代码到远端
./scripts/deploy.sh

# 5. 启动训练
./scripts/remote_train.sh run01

# 6. 查看训练状态
./scripts/remote_status.sh run01       # 概况 + 最后 40 行日志
./scripts/remote_status.sh run01 100   # 最后 100 行日志

# 7. 下载 checkpoint 和 sample
./scripts/fetch_checkpoints.sh run01

# 8. 终止训练（如需）
./scripts/remote_kill.sh run01
```

### 环境变量覆盖

```bash
# 调整 batch size 和 epochs
BATCH_SIZE=16 EPOCHS=500 ./scripts/remote_train.sh run03

# 添加额外参数
EXTRA_ARGS="--lr-g 2e-4 --lr-d 8e-4" ./scripts/remote_train.sh run04
```

### 远端训练进程原理

`remote_train.sh` 在远端执行：
```
nohup python -m src.train ... > train.log 2>&1 &
echo $! > train.pid
```

- 断开 SSH 后进程继续运行（nohup 守护）
- PID 写入 `train.pid` 用于状态检查和中止
- 所有输出重定向到 `train.log`

---

## 监控与评估

### 训练日志

```bash
# 实时查看远端日志
ssh your-server "tail -f ~/ImagePixelNetWork/outputs/run01/train.log"

# 本地查看已下载日志
tail -f outputs/run01/logs/train.log
```

### TensorBoard

```bash
# 本地训练
tensorboard --logdir outputs/run01/logs

# 远端训练（先下载日志）
./scripts/fetch_checkpoints.sh run01
tensorboard --logdir outputs/run01/logs
```

### 关键指标

| 指标 | 健康范围 | 异常信号 |
|------|---------|---------|
| L_content | 下降后稳定在 0.5~2.0 | > 5.0 内容丢失 |
| L_adv (G) | 在 0 附近振荡 | 持续 > 2.0 D 过强 |
| L_d_adv | 在 1~2 振荡 | → 0 D 过强；→ 0 G 过强需检查 |
| L_style_patch | 下降后稳定 | 不下降则纹理未学到 |
| Palette entropy | > 0.5×log(64)≈2.1 | < 1.5 颜色崩塌预警 |

### 样品检查

每 5 个 epoch 保存样品图片到 `outputs/<run>/samples/epochNNNN.png`：

```
[输入原图 1] [输入原图 2] [输入原图 3] [输入原图 4]
[输出结果 1] [输出结果 2] [输出结果 3] [输出结果 4]
```

观察要点：
- Phase 1 (前 50 epochs): 输出模糊但结构正确
- Phase 2 (51-200 epochs): 纹理逐渐"像素化"，颜色趋近 Minecraft 风格
- Phase 3 (201-300 epochs): 边缘锐利，颜色量化明显

---

## 推理导出

训练完成后，使用 checkpoint 对新图像进行推理：

```python
import torch
from PIL import Image
import numpy as np
from src.generator import Generator
from src.palette import PaletteQuantizer

# 加载模型
g = Generator().cuda().eval()
pq = PaletteQuantizer(palette_size=64).cuda().eval()

ckpt = torch.load("outputs/run01/checkpoints/ckpt_epoch0300.pt")
g.load_state_dict(ckpt["g"])
pq.load_state_dict(ckpt["pq"])

# 推理
img = Image.open("your_photo.jpg").convert("RGB").resize((256, 256))
x = torch.from_numpy(np.array(img).astype(np.float32)/255.0).permute(2,0,1).unsqueeze(0).cuda()

with torch.no_grad():
    raw = g(x)
    output = pq(raw, hard=True)  # hard=True: 硬量化

# 保存
out_img = Image.fromarray((output[0].cpu().permute(1,2,0).numpy()*255).astype(np.uint8))
out_img.save("pixel_art_output.png")
```

### 可选后处理：Grid 方块化

```python
def grid_postprocess(img_tensor, grid_size=16):
    """将图像按 grid 分块，每块取众数颜色增强方块感"""
    import torch.nn.functional as F
    B, C, H, W = img_tensor.shape
    # Reshape to grid, take mode per block
    patches = img_tensor.view(B, C, H//grid_size, grid_size, W//grid_size, grid_size)
    patches = patches.permute(0, 2, 4, 1, 3, 5).contiguous()
    patches = patches.view(-1, C, grid_size*grid_size)
    mode_idx = patches.mean(dim=2).argmax(dim=2, keepdim=True)  # simplified; real mode needs more work
    # For proper mode: use torch.unique or quantize
    return img_tensor  # placeholder
```

---

## 常见问题

### Q: 训练崩溃（G loss → ∞, 输出变成纯色）

**原因**: 判别器 D 过强，生成器梯度爆炸。

**解决**:
- 降低 D 学习率：`--lr-d 2e-4`
- 增大 R1 penalty 权重（修改 `src/train.py` 中 `r1_penalty * 10.0` 改为 `* 20.0`）
- 减小 batch size 增加训练噪声

### Q: 输出模糊，没有像素感

**原因**: Phase 2 对抗信号不够强。

**解决**:
- 检查 L_adv 权重是否正常增长到 0.5
- 增加 epochs：`--epochs 400`
- 确认 τ 退火正确（Phase 3 应降至 0.1）

### Q: 颜色崩塌（输出只用少数几种颜色）

**原因**: 调色板使用不均衡。

**解决**:
- 增大 palette loss 权重（修改 `src/train.py` 中 `weights["palette"] = 0.1` 改为 `0.5`）
- 降低学习率
- 重新运行 k-means 初始化调色板

### Q: 输出内容与原图无关

**原因**: Content loss 权重太低或 VGG 特征有问题。

**解决**:
- 确认 content loss 权重为 1.0（Phase 1 应对 content loss 有明显下降）
- 检查 VGG 输入是否做了 ImageNet 归一化（代码已自动处理）
- 增加 Phase 1 的 epochs 数（`if epoch <= 50` 改为 `<= 80`）

### Q: 显存不足 (CUDA OOM)

**解决**:
- 减小 batch size：`--batch-size 4`
- 减小图片尺寸：`--image-size 128`（注意 encoder 最低到 16×16，所以 128 也可用但最小尺度会变）
- 确认 FP16 已启用
- 减少 base_ch：`--base-ch 48`（默认 64）

### Q: 远程训练断连后找不到进程

**解决**:
```bash
# SSH 到远端
ssh your-server
# 查看训练进程
ps aux | grep "src.train"
# 查看日志确认最新 epoch
tail -50 ~/ImagePixelNetWork/outputs/run01/train.log
# 如果没有 PID 文件但进程活着，可以手动记录 PID
echo <PID> > ~/ImagePixelNetWork/outputs/run01/train.pid
```

---

## 项目结构

```
ImagePixelNetWork/
├── README.md                    # 本文件 — 训练教程
├── CLAUDE.md                    # Claude Code 项目指引
├── Docs/                        # 设计文档
│   ├── README.md                # 项目概述
│   ├── architecture.md          # 网络架构
│   ├── dataset.md               # 数据集设计
│   ├── training.md              # 训练策略
│   └── CHANGELOG.md             # 版本记录
├── src/                         # 源代码
│   ├── palette.py               # PaletteQuantizer
│   ├── generator.py             # U-Net Generator
│   ├── discriminator.py         # PatchGAN Discriminator
│   ├── losses.py                # 损失函数 + VGG
│   ├── dataset.py               # 数据加载器
│   └── train.py                 # 训练主程序
├── scripts/                     # 脚本
│   ├── split_collections.py     # 拆分精灵表
│   ├── extract_palette.py       # 提取 k-means 调色板
│   ├── compute_style_stats.py   # 计算 Gram 风格统计
│   ├── remote_config.sh         # 远端服务器配置
│   ├── remote_setup.sh          # 一键配置远端环境
│   ├── deploy.sh                # 部署代码
│   ├── upload_datasets.sh       # 上传数据集
│   ├── remote_train.sh          # 启动远端训练
│   ├── remote_status.sh         # 查看远端状态
│   ├── fetch_checkpoints.sh     # 下载结果
│   └── remote_kill.sh           # 终止远端训练
├── Datasets/                    # 数据集（gitignore）
│   ├── SceneImage/
│   ├── MinecraftImage/
│   ├── palette/
│   └── style/
└── outputs/                     # 训练输出（gitignore）
```

## 引用

如果本项目对你的工作有帮助，欢迎 Star ⭐

```bibtex
@misc{ImagePixelNetWork,
  author = {Garry Chen},
  title = {ImagePixelNetWork: Photo to Minecraft Pixel Art Translation},
  year = {2026},
  url = {https://github.com/GarryChen0402/ImagePixelNetWork},
}
```
