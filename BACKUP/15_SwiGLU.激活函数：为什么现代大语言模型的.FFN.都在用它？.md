# [SwiGLU 激活函数：为什么现代大语言模型的 FFN 都在用它？](https://github.com/lihe/MyGitBlog/issues/15)





在 Transformer 结构中，**前馈网络（FFN）承担了几乎全部的非线性建模能力**。

当模型规模从千万级走向百亿、千亿级后，激活函数不再只是“有没有非线性”的问题，而是：



> **能否在稳定训练的同时，提升表达能力与参数效率。**



SwiGLU，正是在这个背景下成为主流选择的。



------





## **一句话定义**





**SwiGLU 是一种门控激活函数，是 GLU 的改进版本，用 Swish（SiLU）作为门控非线性，通过“值 × 门”的方式增强前馈网络的表达能力，已成为现代大语言模型 FFN 的主流配置。**



关键词一定要说清楚：



- 门控（Gating）
- GLU
- Swish / SiLU
- Transformer FFN





------





## **一、从传统 FFN 说起：问题在哪里？**





以早期 Transformer 常用的 GELU 为例：



$\text{FFN}(x) = W_2 \, \text{GELU}(W_1 x) $



它的问题并不在“不够非线性”，而在于：



- 所有隐藏维度**一视同仁**
- 缺乏显式的**选择机制**
- 表达能力完全依赖单一激活函数





当模型变深、变宽后，这种“统一激活”的方式逐渐成为瓶颈。



------





## **二、GLU：引入门控思想**







### **1. GLU 的基本形式**





GLU（Gated Linear Unit）首次将**门控机制**引入 FFN：



$\text{GLU}(x) = (W_a x) \odot \sigma(W_b x)$



含义非常直观：



- $W_a x$：**值（Value）**
- $\sigma(W_b x)$：**门（Gate，0~1）**
- $\odot$：逐元素乘法（Hadamard 积）





模型不再“被动激活”，而是可以**主动选择哪些维度应该被放大、抑制或关闭**。



------





### **2. 为什么门控有效？**





门控机制本质上引入了**条件计算**：



- 不同 token
- 不同上下文
- 不同层级





都会触发不同的激活模式。



这在序列建模和语言理解中非常关键。



------





## **三、Swish / SiLU：比 Sigmoid 更好的门**





GLU 中最早使用的是 Sigmoid，但它存在明显局限：



- 饱和区梯度接近 0
- 非线性过“硬”





于是 Swish 被引入。





### **Swish 的定义：**





$\text{Swish}(x) = x \cdot \sigma(\beta x)$



当 $\beta = 1$ 时，即为 **SiLU**：



$\text{SiLU}(x) = x \cdot \sigma(x)$



Swish / SiLU 的关键特性：



- 平滑
- 非单调
- 负区间仍保留梯度





这让它成为**理想的门控非线性函数**。

<img width="905" height="67" alt="Image" src="https://github.com/user-attachments/assets/e498cfa7-4d4f-4100-be47-409c427b9d51" />

<img width="450" height="347" alt="Image" src="https://github.com/user-attachments/assets/fdfd2733-cfd8-4a87-b668-af1eba426452" />

------





## **四、SwiGLU 的数学定义**





SwiGLU 就是：

**用 Swish / SiLU 替换 GLU 中的 Sigmoid 门。**



$\text{SwiGLU}(x) = (W_a x) \odot \text{SiLU}(W_b x)$



在 Transformer FFN 中的完整形式：



$\text{FFN}(x) = W_o \big( (W_a x) \odot \text{SiLU}(W_b x) \big)$



结构上可以理解为：



- 两路线性投影
- 一路负责“内容”
- 一路负责“门控强度”





------





## **五、直觉理解（非常重要）**





**SwiGLU = 用一个神经网络，去控制另一个神经网络的输出强度。**



- $W_a x$：提供“信息本身”
- $W_b x$：决定“放多少信息通过”
- SiLU：平滑、连续、可学习的门





相比单一激活函数：



- 更灵活
- 更稳定
- 更适合深层网络





------





## **六、为什么 SwiGLU 比 ReLU / GELU 更好？**







### **1️⃣ 表达能力更强**





- 门控结构 ≈ 特征选择
- 不同维度可以被独立调节
- 更容易建模复杂语义交互





------





### **2️⃣ 梯度传播更稳定**





- ReLU：负区间梯度为 0（死神经元）
- GELU：非门控
- SwiGLU：负区间仍有梯度 + 平滑变化





这在**深层 Transformer**中尤为重要。



------





### **3️⃣ 参数效率更高**





虽然 SwiGLU 多了一次线性投影，但在实践中：



- **同等参数预算 → 更低 perplexity**
- 或 **同等性能 → 更少参数**





这使它在大模型中“性价比极高”。



------





## **七、为什么 FFN 隐藏维度常是 ~2.7d？**





这是很多人看代码时的疑问。



传统 FFN（GELU）：



- hidden_dim = 4d





SwiGLU FFN：



- 有两路投影（$W_a, W_b$）
- 为保持参数量接近：
- hidden_dim ≈ 2.7d





这就是 LLaMA / Qwen / PaLM 中常见配置的来源。



------





## **八、SwiGLU 与常见激活函数对比**



| **激活函数** | **是否门控** | **平滑性** | **表达能力** | **LLM 使用情况** |
| ------------ | ------------ | ---------- | ------------ | ---------------- |
| ReLU         | ❌            | ❌          | 弱           | ❌                |
| GELU         | ❌            | ✅          | 中           | 早期             |
| GLU          | ✅            | ❌          | 高           | 少               |
| **SwiGLU**   | ✅            | ✅          | 高           | **主流**         |



------





## **九、哪些模型在使用 SwiGLU？**





几乎所有主流 LLM：



- PaLM
- LLaMA / LLaMA2 / LLaMA3
- Qwen / Qwen2
- Mistral
- Gemma





可以直接总结一句：



> **现代大语言模型的 FFN，基本都采用 SwiGLU 或其变体。**





