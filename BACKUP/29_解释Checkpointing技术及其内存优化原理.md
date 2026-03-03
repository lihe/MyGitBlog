# [解释Checkpointing技术及其内存优化原理](https://github.com/lihe/MyGitBlog/issues/29)



Checkpointing（又叫 Gradient Checkpointing 或 Activation Checkpointing）本质是：



> 用额外计算换取显存



我会从**原理 → 数学分析 → 复杂度推导 → 为什么是 √n → Transformer 里的具体内存构成 → 和 FlashAttention/ZeRO 的关系 → 工程实践注意点**系统讲清楚。



------





# **一、为什么需要 Checkpointing？**





在训练中，显存主要消耗三部分：



1. **模型参数**
2. **优化器状态（Adam 是 2 倍参数）**
3. **中间激活值（activation）**





对于大模型来说：



> 最大的往往是 activation



尤其是 Transformer：



- batch 大
- seq 长
- 层数多





activation 内存复杂度：



$O(L \cdot B \cdot S \cdot H)$



L = 层数

S = 序列长度



层数一多，activation 线性增长。



------





# **二、标准反向传播为什么要存 activation？**





反向传播根据链式法则：



$\frac{\partial L}{\partial x} = \frac{\partial L}{\partial y} \cdot \frac{\partial y}{\partial x}$



而：



$\frac{\partial y}{\partial x}$



依赖前向时的中间结果。



所以：



> 默认必须保存每层的输出



这就是 activation 内存来源。



------





# **三、Checkpointing 的核心思想**





把网络分成若干段。



前向传播时：



- 只保存“分段的输入”
- 不保存每一层的中间结果





反向传播时：



- 从最近 checkpoint 重新执行前向
- 再计算梯度





换句话说：



> 前向算两次

> 显存减半



------





# **四、数学复杂度分析（关键）**





设网络 L 层。





## **1️⃣ 传统方法**





每层都存 activation：



Memory = O(L)



------





## **2️⃣ 简单分段**





每 K 层做一个 checkpoint：



- 需要存 L/K 个 checkpoint
- 每段回算 K 层





Memory：



$O(L/K)$



Compute：



$O(L \cdot K)$



------





## **3️⃣ 最优策略（√L 定理）**





理论结果：



> 最优分段数是 √L



于是：



Memory：



$O(\sqrt{L})$



Compute：



$O(L \sqrt{L})$



这是经典的：



> Griewank 反向传播时间-空间权衡理论



这也是你图里写的：



O(\sqrt{n})



来源。



------





# **五、直觉理解 √L 为什么出现？**





你可以这样理解：



假设 100 层。



- 每 10 层设一个 checkpoint
- 存 10 个
- 每段回算 10 层





10 × 10 = 100



就是：



$K \cdot \frac{L}{K} = L$



当 K=√L 时达到平衡。



------





# **六、Transformer 里具体省多少？**





Transformer 每层包含：



- QKV projection
- attention matrix
- MLP
- LayerNorm





其中最大的是：



$Attention\ score = O(S^2)$



但 FlashAttention 不存 S²。



真正大的 activation 是：



$B \cdot S \cdot H$



当：



- L=48
- S=4096
- H=4096





activation 占据显存极大。



Checkpoint 可以：



- 显存降 40%~60%
- 训练速度降 10%~30%





------





# **七、PyTorch checkpoint 的机制**





当你写：

```
x = checkpoint(custom_forward, x)
```

内部做的是：



1. forward 时：

   

   - 不保存中间 activation
   - 只记录输入

   

2. backward 时：

   

   - 重新 forward 一遍
   - 再算梯度

   





------





# **八、和 FlashAttention 的关系**





注意：



- FlashAttention 是 IO 优化
- Checkpoint 是 activation 优化





二者可以叠加使用。



很多大模型训练同时使用：



- FlashAttention
- Gradient Checkpointing
- Mixed Precision





------





# **九、和 ZeRO 的区别**





ZeRO 优化的是：



- 参数
- 优化器状态
- 梯度





Checkpoint 优化的是：



- activation





二者优化维度不同。



------





# **十、工程中注意事项**







### **1️⃣ forward 必须纯函数**





不能有：



- dropout randomness（需设置 deterministic）
- inplace 操作
- 非确定性算子





否则回算时会错。



------





### **2️⃣ 不适合所有层**





通常：



- checkpoint Transformer block
- 不 checkpoint embedding





------





### **3️⃣ 会影响速度**





大模型中通常：



- 显存紧张时开启
- 显存够时关闭





------





# **十一、什么时候最值得用？**





✔ 长序列

✔ 层数很多

✔ activation 占主导

✔ 训练超大模型



------





# **十二、面试高质量回答模板**





如果面试问：



> 什么是 Checkpointing？



你可以回答：



> Gradient Checkpointing 是一种通过减少前向传播中保存的中间激活来降低显存使用的技术。它只保存部分层的输入作为检查点，在反向传播时重新计算中间激活值，从而以增加计算量为代价降低内存消耗。理论上可以将 O(L) 的 activation 内存降到 O(√L)，常用于大模型训练。



------





# **十三、更深一层理解**





Checkpoint 本质是：



> 时间-空间权衡（Time-Memory Tradeoff）



它和：



- 递归算法
- 分治策略
- 动态规划





是同一类思想。



------





# **十四、进阶问题（真正高阶）**





为什么：



- 推理阶段几乎不需要 checkpoint？
- 为什么 KV cache 不适合 checkpoint？
- 为什么 activation checkpoint 不会影响模型结果？





# **一、为什么推理阶段几乎不需要 Checkpoint？**







## **1️⃣ 本质原因：推理没有反向传播**





Checkpoint 的存在是为了解决：



> 训练时需要保存 activation 用于反向传播



而推理阶段：



- 不计算梯度
- 不反向传播
- 只做前向





因此：



$\text{activation 用完即可释放}$



不需要长期保存。



------





## **2️⃣ 训练 vs 推理的 activation 生命周期**







### **训练**



```
forward
保存所有 activation
backward
```

activation 必须存到 backward 结束。



------





### **推理**



```
forward
用完即丢
```

没有 backward。



所以：



> 推理 activation 是“流式”的

> 训练 activation 是“堆积”的



------





## **3️⃣ 显存构成对比**







### **训练显存**





- 参数
- 优化器状态（Adam 2倍）
- 梯度
- activation（最大）





------





### **推理显存**





- 参数
- KV cache（主要）
- 当前 token activation（很小）





activation 占比非常低。



所以：



> 推理阶段 activation 不是瓶颈

> 不需要 checkpoint



------





# **二、为什么 KV cache 不适合 Checkpoint？**





这个问题更深。



------





## **1️⃣ KV cache 是什么？**





在自回归生成中：



每生成一个 token：



$K_t, V_t$



都会存起来。



下一步：



$Q_{t+1} K_{1:t}^T$




KV cache 本质是：



> 为了避免重复计算历史 attention



------





## **2️⃣ KV cache 的复杂度**





生成长度 T 时：



- 需要存 T 个 K
- 需要存 T 个 V





内存复杂度：



$O(T \cdot H \cdot L)$



------





## **3️⃣ 为什么不能 checkpoint KV？**





Checkpoint 适用场景：



> 某些值可以在 backward 时重算



而 KV cache 的特性：



- 不是为了反向传播
- 是为了避免重复计算
- 是推理加速核心





如果你对 KV 做 checkpoint：



- 需要重新计算所有历史 token
- 复杂度退化回 O(T²)





这就破坏了：



> 自回归推理 O(T) 复杂度



------





## **4️⃣ 本质区别**



| **项目** | **Activation** | **KV Cache** |
| -------- | -------------- | ------------ |
| 存在阶段 | 训练           | 推理         |
| 作用     | 反向传播       | 加速前向     |
| 可否重算 | 可以           | 不划算       |
| 重算代价 | 小             | 巨大         |



------





# **三、核心本质总结**







### **Checkpoint 的前提**





> 某个值可以低成本重算



activation 在训练中：



- 只需多跑一次 forward
- 成本可接受





KV cache：



- 若重算，需重新算所有历史 token
- 成本指数级增加





所以：



> KV cache 是为了避免重算

> checkpoint 是通过重算节省内存



二者逻辑相反。



------





# **四、从计算图角度看**





训练：

```
x1 → x2 → x3 → x4
```

反向需要：

```
x4 → x3 → x2 → x1
```

activation checkpoint 可以重算。



------



推理：

```
x1 → x2 → x3 → x4
        ↓
   存K,V
```

下一步：

```
x5 需要用 K1,K2,K3,K4
```

如果不存：



就要重新算 x1~x4。



------





# **五、一个形象比喻**





Checkpoint 是：



> 施工时不保留所有脚手架图纸

> 需要时再重画



KV cache 是：



> 已经铺好的铁路轨道

> 下一站必须用



如果你拆掉铁路（checkpoint）：



火车就得重修铁路。



------





# **六、再深一层：复杂度分析**





自回归推理复杂度：



O(T)



若无 KV cache：



$O(T^2)$



所以 KV cache 是时间优化。



Checkpoint 是空间优化。



两者优化方向完全相反。



------





# **七、终极总结**







### **为什么推理几乎不需要 checkpoint？**





因为：



- 没有反向传播
- activation 不需要长期保存
- 显存瓶颈来自 KV cache，而不是 activation





------





### **为什么 KV cache 不适合 checkpoint？**





因为：



- KV cache 的存在就是为了避免重算
- 若 checkpoint 它，就会退化回 O(T²)
- 时间成本远大于内存收益





------





# **八、真正理解之后你会发现**





训练优化关注：



> activation 内存



推理优化关注：



> KV cache 内存 + 带宽



这也是为什么：



- 训练用 Checkpoint
- 推理用 KV cache + FlashAttention
- 超长上下文用分段缓存/滑动窗口



