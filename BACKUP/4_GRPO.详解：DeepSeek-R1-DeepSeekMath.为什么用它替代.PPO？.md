# [GRPO 详解：DeepSeek-R1/DeepSeekMath 为什么用它替代 PPO？](https://github.com/lihe/MyGitBlog/issues/4)






过去一年，“推理模型（Reasoning LLM）”的对齐路线几乎被反复验证了一件事：

**SFT 只是起点，真正把推理能力拉起来的，往往是强化学习阶段。**



而在 DeepSeek-R1 相关讨论里，GRPO（Group Relative Policy Optimization）频繁出现，原因并不玄学——它来自 DeepSeek 之前的工作《DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models》，并在推理/可验证任务上表现出很强的工程合理性。



这篇文章总结一条完整逻辑链：



- 为什么从 PPO 走向 GRPO
- GRPO 的目标函数到底在干嘛
- 结果监督 vs 过程监督的 GRPO
- 迭代 RL（reward model 跟不上怎么办）
- DeepSeekMath 的一个很“硬核”的发现：**RL 为什么有效**





------





## **1. 背景：LLM 对齐训练的常见路线**





在大模型微调里，一个典型流程是：



**预训练 → SFT（监督微调）→ RL（强化学习对齐）**



SFT 让模型“会答题”，但 RL 往往能让模型“更稳定地答对题”。尤其在数学/代码/逻辑推理这种任务中，RL 的提升往往更明显。



问题是：传统 RL 对齐里最常用的 PPO，在 LLM 语境下有几处天然痛点。



------





## **2. 从 PPO 到 GRPO：PPO 在大模型上的痛点是什么？**





PPO（Proximal Policy Optimization）被广泛用在 RLHF/RLAIF 里。它的核心是最大化一个“带截断的替代目标”，并借助优势函数 A 做策略更新。



在 LLM 场景，PPO 常见训练形式还会加入 **KL 约束**，防止策略偏离参考模型（通常是 SFT 模型）太多。





### **PPO 的关键问题**





**(1) 必须训练一个 Value/Critic（值函数）**

值函数通常规模接近策略模型，带来显著的内存与算力开销。更关键的是：critic 训练不稳定、和 policy 不匹配时，训练容易“崩”。



**(2) 工程复杂度高**

Actor + Critic + GAE + KL 控制 + clip，各组件耦合，调参成本高。



**(3) LLM 的奖励结构让 critic 更难学**

很多对齐设置里，奖励模型只在**最后一个 token**（答案结束）给一个整体分数。

这使得“token 级价值估计”在训练上更复杂，critic 的学习信号更弱、更不稳定。



于是一个很现实的想法出现了：



> 很多时候我们并不需要“绝对奖励值”，只需要在同一个 prompt 下**哪条回答更好**。



这就是 GRPO 的出发点。



------





## **3. GRPO 的核心思想：用“组内相对优势”替代 Critic**







### **3.1 一句话定义**





**GRPO 是一种不需要 Critic 的强化学习对齐方法**：

对同一 prompt 采样一组回答，用奖励模型打分，然后用**组内相对差异**构造优势，直接更新策略。



关键词：**group-relative / no critic / 替代 PPO / 更稳定更省**





### **3.2 GRPO 的训练流程**





给定一个问题（prompt）q，从旧策略采样一组输出（group）：



$\{o_1, o_2, \dots, o_G\} \sim \pi_{\theta_{\text{old}}}(\cdot|q)$



用奖励模型（或规则验证器）对每个输出打分：



$r_i = R(q, o_i)$



接下来是 GRPO 的灵魂：构造**组内相对优势**（baseline 不再来自 value model，而来自组平均）：



一种常见形式是：



$A_i = r_i - \frac{1}{G}\sum_{j=1}^{G} r_j$



很多实现里还会做标准化（你材料里也提到）：



$\hat r_i = \frac{r_i - \text{mean}(r)}{\text{std}(r)}$



然后把优势信号用于策略更新，形式上类似 PPO 的“比率 + clip”，但**不需要 Critic/GAE**。



------





## **4. GRPO 的目标函数：它到底优化了什么？**





GRPO 用如下目标对策略进行优化（直觉上是 PPO 风格的 surrogate objective + KL 正则）：



- PPO 里常见做法是把 KL 惩罚放进 reward 里（token-level reward shaping）
- GRPO 的关键区别是：**把 KL 作为损失函数里的正则项**，而不是塞进 reward





直觉很重要：



> PPO：奖励里加 KL 惩罚（reward shaping）

> GRPO：损失里加 KL 正则（regularization）



这让训练更干净，工程上更容易控制。



此外，GRPO 还使用一个无偏估计来计算 KL 散度，并强调该值为正，用于稳定训练。



------





## **5. 结果监督 GRPO：只有“最终答案分”也能训起来**





把 RL 分成了“结果监督”和“过程监督”，这是推理模型训练里非常关键的区分。





### **5.1 结果监督（Outcome Supervision）**





对于同一个问题 q，采样 G 个输出 $o_i$。

奖励模型给每个完整输出一个分数 $r_i$，并进行组内标准化：



$\hat r_i = \frac{r_i - \text{mean}(r)}{\text{std}(r)}$



然后把这个标准化奖励作为该输出**所有 token 的优势**：



$\hat A_{i,t} = \hat r_i$



也就是说：

**整段输出的每个 token，都共享同一个优势信号。**



这在工程上很“粗”，但非常省事，且在很多任务上确实有效。



------





## **6. 过程监督 GRPO：推理任务为什么需要 step-level reward？**





结果监督的问题也很明显：

对于复杂数学推理，仅在最终答案给奖励，信号太稀疏。





### **6.1 过程监督（Process Supervision）**





过程奖励模型会在推理步骤的结束 token（例如每个 step 的结束标记）给分：



对第 i 个输出，共有 $K_i$ 个步骤，步骤结束位置索引为 $\text{index}(j)$，得到奖励序列：



$\{r^i_{\text{index}(1)}, \dots, r^i_{\text{index}(K_i)}\}$



同样做 mean/std 标准化得到 $\tilde r$。



然后每个 token 的优势定义为**从当前 token 往后的 step reward 之和**：



$\hat A_{i,t} = \sum_{\text{index}(j)\ge t} \tilde r^i_{\text{index}(j)}$



直觉上：

token 越早，能“背负”的后续推理步骤越多；

这使得过程监督能更有效地塑形推理轨迹，而不只是塑形最终答案。



------





## **7. 迭代强化学习：reward model 跟不上 policy 怎么办？**




一个非常现实的问题：



> 随着策略模型不断变强，旧的奖励模型可能无法有效监督新的策略分布。



于是 DeepSeekMath 引入了**迭代 RL + GRPO**：



- 每次迭代，用当前策略生成数据
- 构建新的奖励模型训练集
- 用 replay 机制持续训练奖励模型（历史数据占比约 10%）





这本质是在做一件事：



> 让 RM 跟上 policy 的分布漂移，否则 RL 只会“优化一个过时的打分器”。



这也是很多 RLHF 工程里不可避免的循环：**policy 变强 → 分布变了 → RM 失效 → 必须续训 RM**。



------





## **8. DeepSeekMath 的一个关键发现：RL 为什么有效？**






DeepSeekMath 对比了 SFT 与 RL 后模型的 **Pass@K** 与 **Maj@K**：



- RL **显著提升 Maj@K**
- RL **没有提升 Pass@K**





这说明：



> RL 的主要作用不是把“TopK 里的上限”推高（能力上限不变），

> 而是让输出分布更稳健，让正确答案更容易成为“多数/常见输出”。



一句话翻译成大白话：



**RL 更像是在“调分布”，让你更常答对，而不是让你会更多。**



这也解释了为什么推理模型的 RL 往往让表现更“稳”、更“像会推理”。



------





## **9. 统一范式：SFT / DPO / PPO / GRPO 本质上在优化什么？**






所有方法的梯度可以写成一个统一形式，包含三个关键组成部分：



1. 数据源 D（训练数据从哪来）
2. 奖励函数（训练信号来源，如 RM/verifier）
3. 算法 A（如何把数据与奖励变成梯度系数）





在这个范式下：



- **SFT**：用人类挑选的数据监督学习
- **RFT**：对模型采样结果做过滤再监督
- **DPO**：对成对偏好做离线优化（偏监督）
- **Online RFT**：用实时 policy 采样 + 过滤
- **PPO/GRPO**：用实时 policy 采样 + 强化学习更新





> 方法很多，但核心变量其实就三个：数据从哪来、奖励怎么给、算法怎么用奖励更新模型。



------





## **10. GRPO vs PPO vs DPO：面试最容易问的三角对比**







### **10.1 GRPO vs PPO**





- PPO 的优势依赖 Critic / GAE
- GRPO 用组内均值当 baseline，直接构造相对优势
- GRPO 工程更简单、成本更低、稳定性更依赖 reward 排序质量





一句话：

**PPO 学“我比期望好多少”，GRPO 学“我比同组其他答案好多少”。**





### **10.2 GRPO vs DPO**





- DPO：离线偏好数据（chosen/rejected），偏监督学习
- GRPO：在线采样 + reward 打分，是真 RL（但简化了 critic）





一句话：

**DPO 用人类给的对比对；GRPO 用模型自己采样的一组答案做对比。**



------





## **11. GRPO 特别适合哪些任务？**





推理模型偏爱 GRPO 的原因，是它天然适合“可自动打分”的任务：



- 数学（verifier 判对错）
- 代码（单测/编译/静态分析）
- 逻辑推理（规则检查）
- 任何能构建稳定 reward 的任务





当 reward 足够可靠时，组内排序会非常稳定，GRPO 的优势就会被放大。



------





## **12. 局限与工程注意点**





GRPO 也不是银弹，也有风险：



- 需要同 prompt 多采样（group size G）→ 成本来自采样
- group 太小 → 方差大，排序不稳定
- reward 噪声大 → 相对优势抖动，训练不稳





工程上通常会做：



- 合理的 G（group size）选择
- KL 正则强度控制
- clip 稳定训练
- 更可靠的 verifier / PRM 构建与迭代





------





# **总结：为什么 DeepSeek-R1/推理模型会偏爱 GRPO？**





> **GRPO 通过“同 prompt 组内比较”构造相对优势，去掉 critic，极大降低 RL 对齐的工程复杂度与训练成本，并在可验证推理任务上更稳定、更有效。**




> DeepSeekMath 的证据表明，RL 的提升往往来自“让正确答案更常出现”（Maj@K ↑），而不是把能力上限推高（Pass@K 不变）。

> 这也解释了为什么 GRPO 这类稳定分布的 RL 方法，在推理模型时代特别吃香。

