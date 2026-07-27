**prompt：**
我想用以下方式实现：针对stage3的4个transformer层，先不在任何层插入prompt token，让stage2的输出token序列把这四个层跑一遍，得到每个层的输入视觉token（即每个层的中间值），然后在每个层的视觉token中都注入prompt token，并进行一层transformer计算后得到该层的输出，最后在输入的混合token与输出的token序列之间比较差异度，差异度最大的那一组对应的层即为最终的prompt token注入层（因为我认为注入prompt token后，输出token序列的value变化最大的那一层才是最有效的注入层）。选定好prompt token注入层后，stage2的输出token序列会重新走一遍stage3的流程，只是这次会在刚刚选定好的注入层的位置注入prompt token，生成最终的特征
你先要确保理解我的要求是什么，不确定的地方要及时向我询问。同时，向我推荐一个能够很好地比较输入token与输出token序列之间的差异度的机制。

**细节确定：**
差异度比较对象：y_l（第 l 层注入 prompt 后的输出）vs x_{l+1}（干净前向下第 l 层的输出，探测阶段已缓存）——差异纯由 prompt 引起
粒度：逐样本探测选层
CI：探测阶段只用 SI（空间拼接）算差异；选定 l* 后正式前向在该层同时做 SI+CI
对齐注意：y_l 因拼了 prompt token 会多出一行（8×7），比较时只取前 49 个真实 patch token 与 x_{l+1} 的 49 个 patch 对齐，语义 token 本身不参与差异计算

**差异度计算方法：**
推荐 token-wise 余弦距离的均值，理由：SP 全流程用余弦相似度度量特征（原型分类、loss 都是 cosine），选层准则与最终目标同度量最一致；且余弦对 BatchNorm 造成的各层幅度差异不敏感，纯粹衡量"方向改变量"。
建议：主用余弦距离均值；若担心少数 patch 主导，可改用"注意力加权余弦距离"——用该层对语义 token 的注意力权重给各 patch 加权，聚焦 prompt 实际影响的区域。