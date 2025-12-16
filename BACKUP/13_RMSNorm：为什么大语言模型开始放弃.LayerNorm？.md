# [RMSNorm：为什么大语言模型开始放弃 LayerNorm？](https://github.com/lihe/MyGitBlog/issues/13)



## **一句话总结**





**RMSNorm 是一种不做均值中心化、只按均方根缩放特征向量的归一化方法，相比 LayerNorm 计算更简单、参数更少、数值更稳定，因此在深层 Transformer 和大语言模型中更合适。**



关键词：



- 不减均值
- 只控制尺度
- 更高效、更稳定





------





## **一、为什么我们一开始需要 LayerNorm？**





LayerNorm 的初衷很明确：

**稳定深层网络的训练。**



给定一个 token 的隐藏状态向量：



$x \in \mathbb{R}^d$



LayerNorm 做了四件事：



1. 计算均值
2. 计算方差
3. 做标准化（零均值、单位方差）
4. 再通过可学习参数缩放和平移





其形式是：



$\text{LN}(x) = \frac{x - \mu}{\sigma} \cdot \gamma + \beta$



在早期 Transformer 中，这是一个**非常稳妥**的选择。



但问题是：



> **在大模型中，这一步真的“必要”吗？**



------





## **二、RMSNorm 的核心思想：只做一件真正重要的事**





RMSNorm 的设计非常克制。



它只关心一件事：

**控制向量的整体尺度（magnitude）**。





### **数学定义**





给定同样的输入向量：



$x = (x_1, x_2, \dots, x_d)$



先计算均方根（RMS）：



$\text{RMS}(x) = \sqrt{\frac{1}{d} \sum_{i=1}^d x_i^2}$



然后做归一化：



$\text{RMSNorm}(x) = \frac{x}{\text{RMS}(x)} \cdot g$



其中：



- g 是可学习的缩放参数
- **没有减均值**
- **没有偏置项**





------





## **三、RMSNorm vs LayerNorm：真正的结构差异**



<img width="870" height="914" alt="Image" src="https://github.com/user-attachments/assets/d424507e-fd5e-45e2-bae0-852f99e80ce6" />

------





## **四、为什么“不减均值”在 Transformer 里没问题？**





这是理解 RMSNorm 的关键。



在 Transformer 中：



- 信息主要编码在**向量方向**
- Attention 和 FFN 对 **scale 非常敏感**
- 对 **绝对均值并不敏感**
- Residual Connection 会不断引入偏移，均值本身就会漂移





也就是说：



> **强行把均值拉回 0，并不是一个必要约束。**



真正重要的是：



- 不要数值爆炸
- 不要梯度失控
- 不要在深层累积不稳定





RMSNorm 只控制一件事：

**“这个向量不要太大，也不要太小。”**



这恰好是 Transformer 最需要的。



------





## **五、为什么 RMSNorm 在大模型中特别合适？**







### **1️⃣ 计算更省（工程价值极高）**





- 少一次均值计算
- 少一次减法
- 少一个 β 参数
- hidden size 越大，收益越明显





在 4k / 8k hidden size 的 LLM 中，这不是微优化，而是**实打实的吞吐差异**。



------





### **2️⃣ 梯度传播更稳定**





LayerNorm 的均值和方差是动态统计量：



- 深层模型中容易抖动
- 多层叠加后可能出现过度平滑





RMSNorm 不强制中心化：



- 梯度路径更简单
- 数值行为更可控
- 对深层网络更友好





------





### **3️⃣ 与 Pre-LN Transformer 完美契合**





现代 LLM 基本都是：

```
x → Norm → Attention → + x
x → Norm → FFN → + x
```

在这种 **Pre-LN 架构** 中：



- Norm 的作用是“数值保护”
- 而不是“特征重构”





RMSNorm 刚好满足这一角色。



------





## **六、哪些主流模型在用 RMSNorm？**






- LLaMA / LLaMA 2 / LLaMA 3
- Qwen / Qwen2
- Mistral
- Gemma
- GPT-NeoX 系列





------





## **七、RMSNorm 的代价是什么？**








- 理论上表达能力略弱于 LayerNorm
- 在小模型或某些特殊任务中未必最优
- 对“强分布标准化”依赖的场景不合适





但在 **大语言模型** 中：



> **稳定性和效率的收益，远大于这点理论损失。**



------





## **一个好记的直觉类比**





- **LayerNorm**：

  把数据“搬到原点，再缩放”

- **RMSNorm**：

  不管你在哪，只控制“别跑太远”





在 LLM 里，我们更关心的是：



> **“别炸”**，而不是“居中”。



------





## **最终总结**





RMSNorm 是一种简化的归一化方法，只通过均方根对特征向量进行尺度归一化，不进行均值中心化。相比 LayerNorm，它计算更高效、参数更少，并在深层 Transformer 和大语言模型中表现出更好的数值稳定性，因此逐渐成为 LLM 架构中的标准选择。

