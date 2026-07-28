**prompt：**
帮我实现以下内容：在插入语义prompt token后，再在token序列的最前方插入一个经过了一定初始化的token，这个token要设置为可学习的，可以取名叫做multi-modal alignment token，然后随语义prompt token、视觉token一同输入到后续的transformer层中进行前向计算与反向传播。注意，要先加语义prompt token（空间与通道层面都进行）再加multi-modal alignment token。这个方案有两个问题：1、multi-modal alignment token的初始化应该怎样设置？可以参考其他人的可学习prompt token的初始化做法；2、加入这一token后，最终的输出token序列长度会加一，到时候应该如何计算prototype？先不要直接修改代码，要先理解我的要求并明确这两个问题的答案，若有不理解、模糊的地方要向我提出并确定

**问题确定：**
MAT 的初始化方式用哪种？
VPT 的做法，按 fan-in 均匀采样

加入 MAT 后最终特征（用于 prototype）怎么算？
50 个 token 平均，排除语义 prompt 输出

MAT 是共享的可学习参数、不依赖类别文本，理论上 query 分支也可以插入它（语义 prompt 则不行）。MAT 要加到 query 分支吗？
与语义 prompt 绑定出现，保持 query 走原始 forward，行为与原论文一致