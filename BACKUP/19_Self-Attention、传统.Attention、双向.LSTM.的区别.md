# [Self-Attention、传统 Attention、双向 LSTM 的区别](https://github.com/lihe/MyGitBlog/issues/19)





# **一、先讲最基础：Attention 是什么？**





Attention 的核心思想：



> 在做当前预测时，不是只看当前状态，而是“有选择地关注其他位置”。



本质是：



$\text{加权求和}$



------





# **二、传统 Attention（Encoder-Decoder Attention）**





使用场景：



机器翻译等 Encoder-Decoder 结构。



例如：



- Source：英文句子
- Target：中文句子





在翻译某个中文词时：



> 去关注整个英文句子中哪些词重要



------





### **计算过程**





Decoder 当前状态作为 Query

Encoder 所有隐藏状态作为 Key、Value



$\text{Attention}(Q, K, V)$



特点：



- Query 来自 Target
- Key/Value 来自 Source
- 发生在两个序列之间





所以叫：



> Cross Attention（交叉注意力）



------





# **三、Self-Attention 是什么？**





Self-Attention：



> 同一个序列内部做 Attention



也就是：



Target = Source



例如：



句子内部每个词都可以看其他词。



在 Transformer 中：



每个 token 都会生成：



Q, K, V



然后：



$\text{softmax}\left(\frac{QK^T}{\sqrt{d}}\right)V$



------





### **直觉理解**





普通 Attention：



> 我翻译中文时看英文。



Self-Attention：



> 我理解一句话时，词和词之间互相关注。



------





# **四、双向 LSTM 是什么？**





双向 LSTM：



- 一个从左到右
- 一个从右到左
- 最后拼接隐藏状态





优点：



- 能看到前后文





但仍然是：



> 逐步递归计算



------





# **五、核心区别总结**





我们从五个角度对比：



------





## **1️⃣ 信息流方式**



| **机制**       | **信息传播方式** |
| -------------- | ---------------- |
| LSTM           | 顺序递归         |
| 双向 LSTM      | 两个方向递归     |
| Self-Attention | 任意两点直接连接 |



------





## **2️⃣ 长距离依赖**





LSTM：



- 依赖需要经过多步传播
- 距离远 → 梯度衰减





Self-Attention：



- 任意两个 token 直接计算
- 路径长度 = 1





这是最关键差别。



------





## **3️⃣ 并行能力**





LSTM：



- 必须按时间步计算
- 无法并行





Self-Attention：



- 矩阵运算
- 可以完全并行





这就是 Transformer 能用 GPU 高效训练的原因。



------





## **4️⃣ 复杂度**





LSTM：



$O(n)$



Self-Attention：



$O(n^2)$



因为需要两两计算。



------





## **5️⃣ 记忆机制**





LSTM：



- 靠隐藏状态传递
- 逐步压缩信息





Self-Attention：



- 不压缩
- 全局建模





------





# **六、为什么 Self-Attention 更容易建模长距离依赖？**





举例：



句子：



> The book that you gave me yesterday is interesting.



“book”和“is”之间很远。



LSTM：



要经过很多时间步。



Self-Attention：



直接计算两者的相关性。



------





# **七、最核心本质**





LSTM：



> 时间驱动模型（time-driven）



Self-Attention：



> 关系驱动模型（relation-driven）



------





# **八、简单总结对比表**



| **机制**       | **是否跨序列** | **是否并行**       | **长距离依赖** | **路径长度** |
| -------------- | -------------- | ------------------ | -------------- | ------------ |
| Attention      | 是             | 否（依赖 Decoder） | 较好           | 多步         |
| Self-Attention | 否             | 是                 | 非常好         | 1            |
| 双向 LSTM      | 否             | 否                 | 一般           | 多步         |



------





# **九、面试标准回答（可直接说）**





传统 Attention 发生在 Encoder-Decoder 之间，Query 来自 Decoder，Key/Value 来自 Encoder，因此是跨序列注意力。



Self-Attention 是同一序列内部做注意力计算，在 Transformer 中通过 QKV 机制实现，可以直接建模任意两个 token 的关系。



双向 LSTM 通过前向和后向递归来建模上下文，但仍然是顺序计算，长距离依赖需要多步传播，而 Self-Attention 的路径长度为 1，因此更容易捕获长距离依赖并且可以并行计算。



------





# **十、一句话理解**





> LSTM 是“时间传递”

> Attention 是“选择性关注”

> Self-Attention 是“全局建模”

