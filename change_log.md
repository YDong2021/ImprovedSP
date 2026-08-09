## 设计说明

重构注入层选择为单次前向的层内决策网络（LayerSelectNet），替换需要两次前向的 router + val 集双层优化方案：

- 决策网络共享主干（当层 GAP 视觉特征 + 文本特征），4 个独立层头各输出一个决策 logit
- 顺序决策：每层注入前 sigmoid(logit) > 0.5 即注入并终止；前三层都拒绝时第四层兜底强制注入，结构上保证每个样本恰好一层被注入
- STE 直通：前向硬阈值、反向传 sigmoid 梯度，可微且训练/测试逻辑完全一致
- 热启动：层 0/1 偏置 -2、层 2 偏置 +2，初始等价于固定注入 3.2 层
- 删除 router、val 双层优化、gumbel tau 及相关参数；启动标志 `--prompt_layer selective`
- 决策辅助损失：批级均值激活熵（--select_entropy_w，selective 与 multi 共用）防选择/门控坍缩到单层
- multi 模式（`--prompt_layer multi`）：四层软门控注入，w_l=sigmoid(logit_l)，四层皆注入、程度随样本自适应，完全可微（无 STE/兜底/恰好一层约束）；spatial 复用单预留行每层加性刷新，channel 广播加；四层共享 t2i/t2i2 投影；辅助损失仅沿用 select_entropy（作用于门控均值）；监控为每层 gate 均值（train/gate_l{l}）

关键性质
要求	实现
单次前向	决策与注入在同一前向内顺序完成
恰好一层为"是"	顺序"首个过阈值即停" + 第 4 层兜底，构造性保证
可微	STE：前向硬阈值，梯度经 sigmoid 回传到决策头、t2i/se_block、backbone
训练/测试一致	两端同一套硬阈值逻辑，无 Gumbel 采样

一个训练时的观察点：若日志中 fallback 比例持续偏高，说明前三层决策头都过于保守（全部 p<0.5），此时可考虑调小 --decision_warm_bias 之外的干预手段；正常情况下应从初始的层 2 主导逐步演化出分层选择。