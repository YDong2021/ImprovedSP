**prompt：**

我想实现这样一个机制：在元训练与测试过程中，都是在只有处理support sample时才会注入prompt token，这是因为query sample的label未知，无法生成语义prompt token。但我们可以这样做：将所有候选label所生成的prompt token取平均，在处理query sample时，一并注入这个mixed semantic prompt token（也是在空间层面和通道层面都注入），进行后续的transformer运算，而最后的prototype用全部的token序列进行生成。不要直接开始实现，要先理解我的意思，遇到模糊或需要确定的地方要向我提问并明确。



**验证想法：**

视觉token序列的中间量与各label的语义prompt做相似度计算并排序，看看结果怎么样，差别大不大



**变体：**

1. 只在空间层面加入mixed semantic prompt token，而在通道层面不做

2. **加权：**在处理query sample时，stage3的第二层输出的视觉token序列输入stage3的第三层之前，可以先与每一个候选label经过text encoder、projector后的语义token序列做相似度匹配（计算方式可以选用余弦相似度等），然后按照相似度排序，对所有候选label的语义token做加权平均，形成weighted mixed semantic prompt token，然后将其注入到视觉token中（在空间层面和通道层面都注入），进行后续的transformer运算，而最后的prototype用全部的token序列进行生成
3. 加权混合语义提示token只做空间层面的注入，不做通道层面的
4. 把每一个候选label的语义提示token一并注入到视觉token序列中（只在空间层面），而不是取平均或加权平均。