# [RLHF（Reinforcement Learning from Human Feedback，人类反馈强化学习） 的完整流程](https://github.com/lihe/MyGitBlog/issues/46)



RLHF 是 **ChatGPT、Claude、Gemini 等大模型对齐（Alignment）技术的核心**。



我给你从 **背景 → 三阶段流程 → 公式 → 实际训练 → 常见问题 → 新方法** 全面扩展讲清楚。



------





# **一、为什么需要 RLHF**





原始大模型（Pretrained LLM）只是在做：



$P(\text{next token})$



也就是：

```
预测下一个词
```

训练目标是：

```
最大化语言概率
```

但这样训练的模型会出现问题：





### **1 不符合人类偏好**





例如：

```
Q: 如何学习 Python？
```

模型可能回答：

```
Python 是一种编程语言...
```

但人类更喜欢：

```
1. 安装 Python
2. 学习基础语法
3. 做项目
```



------





### **2 不安全**





例如：

```
如何制造炸弹
```

模型可能直接回答。



------





### **3 不礼貌**





例如：

```
你是谁？
```

模型可能乱回答。



------



因此需要：



> **让模型学习人类偏好**



这就是：

```
Alignment
```



------





# **二、RLHF 的核心思想**





核心思想：

```
人类评价模型输出
 ↓
训练奖励模型
 ↓
强化学习优化 LLM
```

流程：

```
Pretrained Model
      ↓
SFT
      ↓
Reward Model
      ↓
RL (PPO)
```



------





# **三、第一阶段：SFT（监督微调）**







### **什么是 SFT**





SFT = **Supervised Fine-Tuning**



使用：

```
人类写的高质量答案
```

训练模型。



------





### **数据格式**





训练数据：

```
Instruction → Response
```

例如：

```
Q: 写一个 Python 排序算法
A: 这是一个 quicksort 实现...
```



------





### **训练目标**





最大化：



$\log P(y|x)$



即：

```
输入问题
输出正确答案
```



------





### **SFT 的作用**





SFT 让模型：



- 学会回答问题
- 学会指令跟随





但问题是：

```
SFT 只学一个答案
```

现实中：

```
答案可能很多
```



------





# **四、第二阶段：Reward Model（奖励模型）**





这是 RLHF 最关键的一步。



------





# **人类偏好数据**





人类会比较：

```
两个回答
```

例如：



问题：

```
如何学习 Python？
```

模型生成两个答案：

```
A: Python 是一种语言...
B: 学 Python 可以从语法开始...
```

人类选择：

```
B 更好
```

得到数据：

```
(B > A)
```



------





# **训练 Reward Model**





奖励模型的作用：

```
预测人类更喜欢哪个回答
```

输入：

```
question + answer
```

输出：

```
score
```

例如：

```
R(A) = 1.2
R(B) = 2.3
```



------





# **Bradley–Terry 模型**





奖励模型训练使用：



$P(y_1 \succ y_2) = \frac{exp(R(y_1))} {exp(R(y_1))+exp(R(y_2))}$



意思是：

```
y1 比 y2 更好的概率
```



------





# **损失函数**





通常使用：

```
cross entropy
```

优化：

```
R(y_good) > R(y_bad)
```



------





# **五、第三阶段：PPO 强化学习**





现在有：

```
LLM
Reward Model
```

接下来：

```
用 RL 优化 LLM
```



------





# **RL 的基本思想**





模型生成：

```
回答
```

Reward Model 给：

```
reward
```

如果回答好：

```
reward 高
```

模型就会：

```
增加这种回答概率
```



------





# **PPO（Proximal Policy Optimization）**





RLHF 通常使用：

```
PPO
```

因为：

```
稳定
```



------





# **PPO 目标函数**





图中公式：



$\max_{\pi} E[R(y)-\beta KL(\pi||\pi_{SFT})]$



含义：





### **第一部分**





R(y)



奖励模型给的分数。



------





### **第二部分**





$KL(\pi||\pi_{SFT})$



限制模型不要偏离 SFT 太远。



------





### **为什么要 KL**





如果没有 KL：



模型可能学会：

```
reward hacking
```

例如：

```
一直说“谢谢你的问题”
```

骗奖励。



------





# **六、RLHF 完整流程**





完整 pipeline：

```
1 预训练
   ↓
2 SFT
   ↓
3 生成多个回答
   ↓
4 人类排序
   ↓
5 训练 Reward Model
   ↓
6 PPO 优化 LLM
```



------





# **七、RLHF 的难点**







### **1 数据昂贵**





需要：

```
人类标注
```

成本非常高。



------





### **2 奖励黑客（Reward Hacking）**





模型可能：

```
欺骗奖励模型
```

例如：

```
长回答更高分
```



------





### **3 Reward Model 偏差**





奖励模型不一定完全正确。



------





# **八、现代替代方法**





现在很多公司逐渐减少 PPO。



------





# **1 DPO（Direct Preference Optimization）**





DPO 不需要：

```
Reward Model
```

直接用：

```
偏好数据
```

优化模型。



------



目标：



$\log \sigma(\beta(\log \pi(y^+)-\log \pi(y^-)))$



优点：



- 更简单
- 更稳定





------





# **2 RLAIF**





AI 来做标注：

```
AI Feedback
```

减少人类成本。



------





# **九、真实公司流程**





例如 OpenAI：

```
GPT Pretrain
   ↓
SFT
   ↓
Human ranking
   ↓
Reward model
   ↓
PPO
   ↓
ChatGPT
```



------





# **十、面试标准回答**





如果面试官问：



**解释 RLHF**



可以这样回答：



> RLHF（Reinforcement Learning from Human Feedback）是一种通过人类反馈对齐大模型行为的方法。首先使用人工标注的高质量问答数据进行监督微调（SFT），让模型具备基本的指令跟随能力；然后通过人类对模型输出进行排序，训练奖励模型（Reward Model）来预测人类偏好；最后利用强化学习算法（如 PPO）优化策略模型，使模型生成的回答能够获得更高的奖励，同时通过 KL 散度约束避免模型偏离原始 SFT 模型过远。



------



如果你愿意，我可以再给你讲一个 **RLHF 中最容易被问到的深度问题**：



**为什么 RLHF 要用 PPO，而不是普通策略梯度？**



这个问题其实涉及 **策略稳定性问题**，很多人答不上来。

---

https://zhuanlan.zhihu.com/p/677607581