# Instance-Level Caption Prompt 使用说明

## 概述

将原始 SP 的 class-level 静态文本模板（`"A photo of {label}"`）替换为 LLaVA 为每张图片生成的 instance-level 描述，经 CLIP 编码后作为 per-sample 语义提示注入 Visformer。

## 环境依赖

在原有依赖基础上，新增：

```bash
pip install transformers accelerate
```

LLaVA 模型权重（`llava-hf/llava-1.5-7b-hf`）会在首次运行时自动从 HuggingFace 下载。

---

## 完整流程

### Step 1：离线生成图片描述

对数据集的每个 split 分别运行：

```bash
# miniImageNet - train split (base classes)
python generate_captions.py --dataset miniImageNet --dataset_folder miniImagenet --split train

# miniImageNet - test split (novel classes)
python generate_captions.py --dataset miniImageNet --dataset_folder miniImagenet --split test

# miniImageNet - val split (如需在 val 上验证)
python generate_captions.py --dataset miniImageNet --dataset_folder miniImagenet --split val
```

**参数说明：**

| 参数 | 说明 |
|------|------|
| `--dataset` | 数据集名称，用于输出文件命名 |
| `--dataset_folder` | `./dataset/` 下的实际文件夹名（如 `miniImagenet`），默认同 `--dataset` |
| `--split` | `train` / `val` / `test`，分别对应 `base` / `val` / `novel` 子目录 |
| `--model_id` | LLaVA 模型路径，默认 `llava-hf/llava-1.5-7b-hf` |
| `--max_tokens` | 最大生成 token 数，默认 77（对齐 CLIP 上下文窗口） |

**输出：** `data/captions/{dataset}_{split}_captions.json`

**其他数据集示例：**

```bash
# tieredImageNet
python generate_captions.py --dataset tieredImageNet --dataset_folder tieredImageNet --split train
python generate_captions.py --dataset tieredImageNet --dataset_folder tieredImageNet --split test

# CIFAR-FS
python generate_captions.py --dataset CIFAR-FS --dataset_folder cifar100 --split train
python generate_captions.py --dataset CIFAR-FS --dataset_folder cifar100 --split test

# FC100
python generate_captions.py --dataset FC100 --dataset_folder FC100 --split train
python generate_captions.py --dataset FC100 --dataset_folder FC100 --split test
```

---

### Step 2：CLIP 编码描述文本

```bash
# miniImageNet
python encode_captions.py --dataset miniImageNet --split train
python encode_captions.py --dataset miniImageNet --split test
```

**参数说明：**

| 参数 | 说明 |
|------|------|
| `--dataset` | 与 Step 1 一致 |
| `--split` | 与 Step 1 一致 |
| `--gpu` | GPU 编号，默认 0 |
| `--batch_size` | 编码批次大小，默认 256 |

**输出：** `data/captions/{dataset}_{split}_clip_features.pt`，shape `[N_samples, 512]`

> 注意：使用 CLIP 完整 77-token 上下文，无截断。已包含 eqnorm 归一化。

---

### Step 3：训练

```bash
# miniImageNet 1-shot
python train_vit_sp.py --gpu 0 --dataset miniImageNet --exp sp_caption \
    --use_caption \
    --init checkpoint/miniImageNet/visformer-t/pre-train/checkpoint_epoch_800.pth

# miniImageNet 5-shot
python train_vit_sp.py --gpu 0 --dataset miniImageNet --exp sp_caption_5shot \
    --shot 5 --use_caption \
    --init checkpoint/miniImageNet/visformer-t/pre-train/checkpoint_epoch_800.pth
```

关键新增参数：`--use_caption`（启用 per-image caption 模式）

其余参数与原始 SP 完全一致（`--stage`, `--prompt_mode`, `--t`, `--lr` 等）。

---

### Step 4：测试

```bash
# 1-shot prototype
python train_vit_sp.py --gpu 0 --dataset miniImageNet --exp test_caption \
    --use_caption --test --episodes 2000 \
    --resume checkpoint/miniImageNet/visformer-t/sp_caption/checkpoint_epoch_best.pth

# 5-shot + LR classifier + augmentation
python train_vit_sp.py --gpu 0 --dataset miniImageNet --exp test_caption_5shot \
    --shot 5 --use_caption --test --episodes 2000 \
    --test_classifier fc --aug_support 10 \
    --resume checkpoint/miniImageNet/visformer-t/sp_caption_5shot/checkpoint_epoch_best.pth
```

---

## 向后兼容

不加 `--use_caption` 时，行为与原始 SP 完全一致（class-level 模板 + text_length 截断）。

---

## 文件结构

```
data/captions/
├── miniImageNet_train_captions.json      # {index: caption_string}
├── miniImageNet_test_captions.json
├── miniImageNet_train_clip_features.pt   # [N_train, 512]
└── miniImageNet_test_clip_features.pt    # [N_test, 512]
```

## 注意事项

1. **索引对齐**：captions.json 和 clip_features.pt 的索引与 `torchvision.datasets.ImageFolder` 的 `samples` 顺序严格一致
2. **显存需求**：LLaVA-7B 需约 14GB 显存（FP16）；若显存不足可使用 4-bit 量化加载
3. **生成质量**：Prompt 中包含类别标签提示，引导 LLaVA 生成更聚焦的描述
