# 动态 Prompt 注入层选择机制设计

## 背景

当前实现（Semantic Prompt, CVPR 2023）将语义 prompt 固定注入在 Visformer stage3 的某一层（默认 `--stage 3.2`，即 layer3-2）。论文 §5.3.2 表明：不同数据集的最优注入层略有差异，且 stage3 的 4 层（3.0 / 3.1 / 3.2 / 3.3）均有效。

**目标**：设计一个按类别动态选择注入层的机制，在 stage3 的 4 个候选层中为每个类别选出最优注入位置。

**可行性基础**：stage3 四层特征维度相同（7×7, C=384），现有投影层 `t2i`（spatial）与 `t2i2`（channel）可被 4 个候选位置共享，无需额外投影参数。

---

## 方案 A：Gumbel-Softmax 路由器（离散选层）

### 核心思想

引入一个轻量门控网络（router），根据类别文本特征输出 4 个候选层的选择分布，通过 Gumbel-Softmax 解决离散选层的不可微问题。训练时软加权注入，测试时 argmax 硬选一层。

### Router 结构

```python
self.router = nn.Sequential(
    nn.Linear(512, 128),   # 输入：冻结 CLIP 文本编码器输出 g(y_text)
    nn.ReLU(),
    nn.Linear(128, 4),     # 输出：4 个 logits，对应 stage3 的 4 个注入位置
)
```

- 参数量约 66K（相对整网 10M 可忽略）
- 输入只用文本特征 → 同一类别的选层结果稳定一致（类别级决策）
- 本质是 MoE 中的 gating network，"专家"是同一 prompt 的 4 个注入位置

### 工作流程

1. `logits = router(g(y_text))`
2. 训练：`w = gumbel_softmax(logits, τ)`，τ 从 5 退火至 0.5，后期可用 straight-through
3. 测试：`w = one_hot(argmax(logits))`，只在选中层注入

### 注入方式（改造 `forward_with_semantic_prompt_channel` 的 stage3 循环）

| 机制 | 可微性 | 做法 |
|---|---|---|
| CI（通道注入） | 天然可微（加法） | 每层都计算调制向量 β_l，按 `x = x + w_l * β_l` 加权注入 |
| SI（空间注入） | 拼接不可直接加权 | 4 层都拼 prompt token，但 token 内容为 `w_l * prompt`；τ 退火后 w 趋近 one-hot，等效单层注入；测试时仅在 argmax 层拼真 token |

### 训练技巧

- **Warm start**：router 输出层 bias 初始化偏向 3.2（已知最优默认层），避免早期乱选破坏预训练特征
- **防塌缩**：加熵正则或 load-balancing loss，防止路由早期塌缩到单一层
- **温度退火**：τ: 5 → 0.5，保证训练/测试一致性
- **学习率分组**：router 参数与 `t2i`/`t2i2` 同放高 lr 组（5e-4），backbone 保持 1e-6
- **可选扩展**：router 输入拼接 stage2 输出的全局池化特征，变为 instance-aware（但会失去类别级一致性）

---

## 方案 B：全层软加权（连续混合，无离散选择）

### 核心思想

放弃"只选一层"，stage3 的 4 层**全部注入**，但用 router 输出的 softmax 权重对各层注入强度加权：

```python
w = softmax(router(g(y_text)))   # 无 Gumbel 采样、无温度退火
# 第 l 层：CI 按 w_l 缩放 β_l；SI 拼接 w_l * prompt token
```

### 与方案 A 的关系

- 实现最简单、训练最稳（无采样噪声、无退火调参）
- 训练/测试行为完全一致（都是软加权）
- 作为方案 A 的 baseline：
  - 若 A 学出的分布本身接近 one-hot → 两者等价，A 的离散化收益有限
  - 若 B 明显更好 → 说明"多层混合注入"优于"单层选择"，本身即有价值的结论

---

## 方案对比

| 维度 | 方案 A（Gumbel 离散选层） | 方案 B（全层软加权） |
|---|---|---|
| 决策形式 | 每类硬选 1 层（测试时） | 每类 4 层加权混合 |
| 可微性处理 | Gumbel-Softmax + 温度退火 | softmax，天然可微 |
| 训练稳定性 | 需退火/正则调参 | 稳定 |
| 推理开销 | 与原实现相同（单层注入） | 4 层均注入，略高 |
| 可解释性 | 强（明确的类别→层映射） | 中（权重分布） |
| 实现工作量 | 中 | 低（约半天） |

---

## 落地改动点

1. **[visformer.py](visformer.py)**：stage3 循环从"计数器精确匹配 `args.stage`"改为接收 4 维 gate 向量 `w`，逐层按 `w_l` 执行 CI 加权 / SI 缩放拼接
2. **[train_vit_sp.py](train_vit_sp.py)**：
   - 创建 `student.router`，参数加入 `optim_params_id` 高 lr 组
   - 新增 `--prompt_layer {fixed, dynamic}` 开关，保留原固定模式做对照
   - 方案 A 另需 `--gumbel_tau_start/end` 退火参数与熵正则系数
3. **评估协议**：
   - 基线：固定注入 3.0 / 3.1 / 3.2 / 3.3 四组 + 方案 B
   - 指标：miniImageNet / CIFAR-FS 等 5-way 1-shot / 5-shot 准确率
   - 分析：输出每类选层分布，检验是否学到有意义的类别差异（如细粒度类偏高层、粗粒度类偏低层）

## 实施顺序建议

先跑方案 B 确认动态加权有收益（工作量小、风险低），再引入方案 A 的 Gumbel 离散化对比"混合注入 vs 单层选择"。
