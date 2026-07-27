# 1.1

原始semantic prompt

# 1.2 

改动训练时的报错

[add_disable_beta_transforms_warning](https://github.com/YDong2021/ImprovedSP/commit/d046e32e9f0abe8eb0eea198b10e410dcd9fca2e)

## 1.2.1.1

动态选择插入层机制。实现方式是我本人的：比较有prompt注入与无prompt注入时的差异，选差异最大的一层为最终注入层。硬截断，不可微。