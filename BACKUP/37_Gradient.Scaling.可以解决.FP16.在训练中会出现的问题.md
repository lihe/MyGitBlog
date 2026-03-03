# [Gradient Scaling 可以解决 FP16 在训练中会出现的问题](https://github.com/lihe/MyGitBlog/issues/37)




# **一、FP16 为什么容易出问题？**







## **1️⃣ FP16 的数值范围**





FP16（IEEE half precision）：



- 1 位符号
- 5 位指数
- 10 位尾数





可表示范围大约：



$[6 \times 10^{-8}, \; 6.5 \times 10^4]$



问题来了：



> 深度学习中的梯度经常小于 1e-7



例如：



$3 \times 10^{-9}$



在 FP16 中：



会被直接舍入为 **0**



这叫：



> underflow（下溢）



------





# **二、为什么梯度容易很小？**





因为梯度是链式法则连乘：



$\frac{\partial L}{\partial x} = \frac{\partial L}{\partial y} \cdot \frac{\partial y}{\partial x}$



多层网络：



$g = g_L \times W_L \times W_{L-1} \times \dots$



每乘一次：



都可能 < 1



最后：



梯度可能极小。



------





# **三、underflow 会导致什么？**





梯度被变成 0：



- 参数不更新
- 训练停滞
- loss 不下降





这在大模型中非常常见。



------





# **四、Gradient Scaling 的核心思想**





关键观察：



> 梯度是线性的。



如果：



L' = sL



则：



$\nabla L' = s \nabla L$



也就是说：



放大 loss → 梯度等比例放大。



------





# **五、完整流程解释**





假设 scale = 1000



------





## **Step 1️⃣：放大 loss**





L' = 1000L



------





## **Step 2️⃣：反向传播**





计算得到：



g' = 1000g



此时：



小梯度被放大到 FP16 能表示的范围。



------





## **Step 3️⃣：更新前缩回**





更新时：



$g = \frac{g'}{1000}$



最终更新量：



$\theta = \theta - \eta g$



与原始完全一致。



------





# **六、为什么不会影响训练结果？**





因为：



$\nabla (sL) = s\nabla L$



最后除以 s。



整体等价于：



$(s / s) = 1$



数学上完全等价。



------





# **七、为什么只放大 loss，不放大梯度？**





因为：



梯度来自 loss。



放大 loss：



自动放大所有中间梯度。



不用手动干预每层。



------





# **八、为什么不能一直用很大的 scale？**





因为会 overflow。



如果：



scale 太大：



梯度可能超过 FP16 上限：



$\> 6.5 \times 10^4$



变成：



inf



训练直接崩溃。



------





# **九、Dynamic Loss Scaling**





现代框架（PyTorch AMP）做：



- 先用大 scale
- 如果检测到 inf/NaN
- 自动减半 scale





这叫：



> 动态梯度缩放



------





# **十、完整数学直观总结**





真实更新：



$\theta = \theta - \eta \nabla L$



缩放训练：



$\theta = \theta - \eta \frac{1}{s} \nabla (sL)$



等价。



------





# **十一、为什么混合精度能加速？**





因为：



- FP16 计算更快
- 显存减少 50%
- 通信数据减少 50%





但：



权重通常保留 FP32 master copy。



------





# **十二、为什么权重不用纯 FP16？**





因为：



- 参数更新累积误差
- 数值不稳定





常见做法：



- forward/backward 用 FP16
- optimizer state 用 FP32





------





# **十三、一个简单类比**





想象你有一个极小的数：



0.000000003



FP16 看不到。



你把它放大 1000 倍：



0.000003



现在看得到。



算完再缩回。



结果一样。



------





# **十四、总结一句话**





Gradient Scaling 的本质是：



> 利用梯度线性性质，临时放大 loss 来避免 FP16 下溢，再在更新前缩回，数学上完全等价，但数值上更稳定。



------





# **十五、再往深一点（高级理解）**





为什么 bfloat16 很少需要 scaling？



因为：



- 指数位 8 位（和 FP32 一样）
- 数值范围大
- 不容易 underflow





这就是为什么 TPU 更喜欢 bfloat16。



------

