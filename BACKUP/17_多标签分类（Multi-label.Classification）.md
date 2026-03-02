# [多标签分类（Multi-label Classification）](https://github.com/lihe/MyGitBlog/issues/17)






# **一、多标签分类的 Loss 设计**







## **1️⃣ 什么是多标签分类？**





- 一个样本可以对应 **多个标签**
- 标签总类别数固定（比如 10 类）
- 每个样本的标签数量不固定





例如：

类别总数 10 类

某张图属于 0、3、9 类

编码为：

```
1 0 0 1 0 0 0 0 0 1
```

注意：



- 不能用 softmax
- 不能用普通 one-hot（只能单标签）





------





## **2️⃣ 为什么用 Sigmoid + Binary Cross Entropy？**





多标签问题本质是：



> 把每个类别看成一个独立的二分类问题



每个类别单独判断“是否属于该类”。



输出层：

```
σ(l_k) = sigmoid(logit_k)
```

损失函数：



$L = - \sum_k \left[y_k \log \sigma(l_k) + (1-y_k)\log(1-\sigma(l_k))\right]$



本质理解：



- 每个标签做一次二分类
- 所有标签 loss 求和 / 平均
- 等价于多个 independent binary classification





------





## **面试官真正想听你说：**





- 多标签不能用 softmax（会强制归一）
- 用 sigmoid 让每个标签独立
- 用 BCE 计算每个标签的损失





------





# **二、多标签的评价指标**





这是这题的重点考察部分。





## **1️⃣ Subset Accuracy（严格准确率）**





只有：



> 预测的标签集合 和 真实标签集合 完全一致



才算对。



公式：



$\frac{1}{p}\sum 1(h(x^i)=y^i)$



特点：



- 非常严格
- 工业界很少用
- 标签多时几乎为 0





------





## **2️⃣ Hamming Loss**





衡量：



> 错误标签的比例



公式：



$\frac{1}{p}\sum \frac{1}{q}|h(x^i)\Delta y^i|$



Δ 是对称差



理解：



- 每个标签单独看
- 错一个就加 1
- 适合标签很多的场景





优点：更稳定



------





## **3️⃣ One-error**





看：



> 预测分数最高的标签 是否在真实标签里



值越小越好。



衡量排序能力。



------





## **4️⃣ Coverage**





看：



> 排序后的标签列表需要走多远才能覆盖所有真实标签



衡量排序质量。



------





# **三、类别不平衡问题**





这是第三问，考察你是否理解工程问题。



------





## **1️⃣ Threshold Moving（阈值移动）**





默认：

```
y > 0.5 判正
```

不平衡时：



根据正负样本比例调整：



$\frac{y}{1-y} > \frac{m^+}{m^-}$



直觉：



正样本少 → 降低判正阈值



------





## **2️⃣ 欠采样（Undersampling）**





减少多数类样本



缺点：丢信息



------





## **3️⃣ 过采样（Oversampling）**





增加少数类样本



缺点：可能过拟合



------





## **4️⃣ Focal Loss**





目标检测里常用：



$FL(p) = -(1-p)^\gamma \log p$



作用：



- 降低易分类样本权重
- 聚焦难样本
- 解决正负样本极度不平衡





------





# **面试总结版（你可以这样回答）**





这道题考察三点：



第一，多标签分类的 loss 设计



- 不能用 softmax
- 用 sigmoid + BCE
- 每个标签独立二分类





第二，多标签的评价指标



- Subset Accuracy（严格匹配）
- Hamming Loss（标签级别错误率）
- One-error（top1 是否正确）
- Coverage（排序覆盖度）





第三，类别不平衡解决方案



- 阈值移动
- 过采样 / 欠采样
- Focal Loss





------





# **如果面试官继续追问**





你可以补充：



- 多标签也可以用 label smoothing
- 可以使用 class weight
- 可以用 Asymmetric Loss
- 可以使用 macro / micro F1
- 可以做 per-class threshold tuning





------





# **一句话总结这题的本质**





> 多标签 = 多个独立二分类 + 排序问题 + 不平衡优化


---

https://blog.csdn.net/tsyccnh/article/details/79163834