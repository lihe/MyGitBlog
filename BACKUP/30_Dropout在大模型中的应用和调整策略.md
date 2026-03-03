# [Dropout在大模型中的应用和调整策略](https://github.com/lihe/MyGitBlog/issues/30)




# **一、Dropout 的本质是什么？**





Dropout 的核心思想：



> 训练时随机“丢弃”一部分神经元

> 推理时使用完整网络



数学形式：



设某层输出：



h = f(x)



Dropout 后：



$\tilde{h} = \frac{m \odot h}{1-p}$



其中：



- $m \sim Bernoulli(1-p)$
- p 是 dropout 概率
- 除以 1-p 是为了保持期望不变





------





# **二、Dropout 的理论解释**







### **1️⃣ 防止过拟合**





它等价于：



> 训练时采样不同子网络

> 推理时做模型平均



类似：



> 近似做 ensemble



------





### **2️⃣ 降低 co-adaptation**





没有 dropout 时：



某些神经元可能高度依赖其他神经元。



Dropout 强迫：



> 每个神经元必须单独“有用”



------





### **3️⃣ 从贝叶斯角度看**





Dropout 可以近似为：



> 对权重做变分贝叶斯推断



这是 Gal & Ghahramani 的经典解释。



------





# **三、在 Transformer 中的 Dropout 位置**





Transformer 常见 4 个 dropout：



1️⃣ attention score 后

2️⃣ attention 输出后

3️⃣ FFN 输出后

4️⃣ embedding 后



BERT 典型配置：

```
hidden_dropout_prob=0.1
attention_probs_dropout_prob=0.1
```



------





# **四、不同位置的影响**







### **1️⃣ Attention probs dropout**





softmax(QK^T)



随机丢弃注意力权重。



效果：



- 防止 attention 过拟合某些 token





------





### **2️⃣ FFN dropout**





通常效果更明显。



原因：



- FFN 参数占 2/3
- 更容易过拟合





------





### **3️⃣ Embedding dropout**





较弱正则化。



------





# **五、大模型中 Dropout 的变化趋势**





这是关键点。



你会发现：



> GPT-3、LLaMA、PaLM 等大模型几乎不用 Dropout



为什么？



------





## **1️⃣ 数据规模极大**





当数据量非常大时：



> 数据本身就是正则化



过拟合风险小。



------





## **2️⃣ LayerNorm 已提供稳定性**





Transformer 的：



- Residual
- LayerNorm





已经降低过拟合。



------





## **3️⃣ Dropout 会影响收敛速度**





Dropout 增加噪声：



- 训练变慢
- 大模型训练成本高





所以预训练阶段常关掉或设为 0。



------





# **六、什么时候仍然用 Dropout？**





✔ 小模型

✔ 数据少

✔ 微调阶段

✔ 下游任务



------





# **七、动态 Dropout 策略**





训练初期：



- dropout 高一点（防止早期过拟合）





训练后期：



- 逐渐降低（让模型充分拟合）





类似学习率 schedule。



------





# **八、替代方案**







### **1️⃣ Stochastic Depth**





随机跳过整层：



$y = x + mF(x)$



m=0 时整层跳过。



适用于：



- 很深网络
- ViT





------





### **2️⃣ DropPath**





在 ViT 中常用：



随机丢弃路径而不是单个神经元。



------





### **3️⃣ Label Smoothing**





另一种正则方式。



------





### **4️⃣ Weight Decay**





L2 正则。



------





# **九、Dropout 和梯度的关系**





Dropout 会：



- 增加梯度方差
- 降低梯度范数稳定性





所以：



大模型训练时：



> dropout 太大会 destabilize



------





# **十、预训练 vs 微调**



| **阶段** | **Dropout** |
| -------- | ----------- |
| 预训练   | 很小甚至 0  |
| 微调     | 常 0.1~0.3  |



------





# **十一、一个常见误区**





❌ Dropout 越大越好

错。



过大 dropout：



- 欠拟合
- loss 不下降





------





# **十二、工程经验值**



| **模型规模** | **推荐 dropout** |
| ------------ | ---------------- |
| 小模型       | 0.3              |
| 中等模型     | 0.1              |
| 大模型       | 0~0.1            |
| 超大模型     | 0                |



------





# **十三、面试回答模板**





如果面试问：



> Dropout 在大模型中的作用？



你可以答：



> Dropout 通过随机丢弃神经元防止过拟合，相当于对模型做隐式 ensemble。在 Transformer 中通常用于 attention 权重和 FFN 层。但在大规模预训练模型中，由于数据规模巨大、LayerNorm 和 residual 提供稳定性，Dropout 常被降低甚至关闭，以避免增加训练噪声和影响收敛速度。微调阶段则常重新开启。



------





# **十四、一个更深层问题**





为什么：



- 大模型不用 Dropout 但仍不过拟合？
- 为什么 L2 正则仍然重要？
- 为什么 Dropout 会影响 calibration？

