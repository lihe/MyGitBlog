# [P-Tuning 与 P-Tuning v2：参数高效 Prompt 微调方法解析](https://github.com/lihe/MyGitBlog/issues/5)



## **一、方法提出背景**





在大语言模型的下游任务适配中，Prompt 的构造方式对模型性能具有决定性影响。以 GPT-3 为代表的 In-Context Learning 依赖人工设计的离散模板，但这种方式存在明显局限：



- 模板对措辞、顺序和示例极其敏感
- 轻微修改即可导致性能大幅波动
- 离散 token 的自动化搜索成本高，且难以保证稳定最优





同时，全参数微调虽然效果稳定，但在大模型场景下带来了显著的显存、算力与维护成本，并伴随潜在的灾难性遗忘风险。



在这一背景下，P-Tuning 被提出，其核心目标是在**冻结模型主体参数的前提下，通过可学习 Prompt 实现稳定、高效的任务适配**。



------





## **二、P-Tuning 的核心思想**





P-Tuning 将传统 Prompt 中的离散 token 替换为一组**连续可微的虚拟 token（Virtual Tokens）**，并直接在 embedding 空间中进行优化。



核心特征包括：



- Prompt 不再是自然语言文本，而是可训练向量
- Prompt 作为模型输入的一部分参与前向传播
- 模型主体参数保持冻结，仅优化 Prompt 相关参数





这一设计使 Prompt 构造从“人工经验问题”转化为“连续优化问题”。



------





## **三、P-Tuning 的技术原理**







### **1. 连续 Prompt 表示**





P-Tuning 将人工模板中的真实 token 替换为一组虚拟 token，这些 token 以 embedding 的形式插入输入序列中。

虚拟 token 的插入位置并不固定，可以是前缀，也可以位于输入序列的其他位置。





### **2. Prompt Encoder 设计**





由于预训练语言模型的 embedding 空间高度离散，若直接随机初始化虚拟 token，优化过程容易陷入局部最优。



为此，P-Tuning 引入 Prompt Encoder，对虚拟 token 进行结构化建模：



- 使用 LSTM + MLP 对 Prompt embedding 进行编码
- 显式建模 Prompt token 之间的相关性
- 提升收敛速度与优化稳定性





实验结果表明，引入 Prompt Encoder 后，P-Tuning 在多个任务上可以达到甚至超过全参数微调的效果。



------





## **四、P-Tuning 的实验现象与结论**





在对比实验中，P-Tuning 展现出若干值得注意的现象：



- 在相同参数规模下，全参数微调时 BERT 在 NLU 任务中通常优于 GPT
- 在 P-Tuning 设定下，GPT 在多个任务上可反超 BERT
- P-Tuning 在部分任务中可达到与全参数微调一致的性能





这些结果表明，**模型能力并不完全由参数更新方式决定，输入空间的可学习结构同样具有重要影响**。



------





## **五、P-Tuning v2 的提出动机**





尽管 P-Tuning 在多个任务中取得了良好效果，但其仍存在明显局限：





### **1. 规模泛化能力不足**





- 在大模型（>10B）上效果接近全参微调
- 在中小模型（100M–1B）上性能明显下降







### **2. 任务泛化能力不足**





- 在部分 NLU 分类任务中表现良好
- 在序列标注等结构化任务上效果不稳定







### **3. 提示深度受限**





- Prompt 仅作用于输入 embedding 层
- 对深层 Transformer 表征的影响较为间接





为解决上述问题，P-Tuning v2 被提出。



------





## **六、P-Tuning v2 的核心改进**





P-Tuning v2（论文 *P-Tuning v2: Prompt Tuning Can Be Comparable to Fine-tuning Universally Across Scales and Tasks*）引入了**深度 Prompt 优化机制**。





### **1. 多层 Prompt 注入**





- 在 Transformer 的每一层引入 Prompt token
- Prompt 直接参与中间层表示计算
- 对模型预测产生更直接的影响







### **2. 参数规模提升但仍高效**





- 可学习参数比例从约 0.01% 提升至 0.1%–3%
- 相较全参数微调仍具显著参数效率优势







### **3. 结构设计上的简化**





- 移除 P-Tuning v1 中的重参数化 Prompt Encoder
- 直接将 Prompt 作为可学习参数
- 在小模型场景下表现更稳定





------





## **七、任务适配与训练策略**





P-Tuning v2 在工程实现中引入了多项关键策略：



- 针对不同任务设置不同 Prompt Length

  

  - 简单分类任务：较短 Prompt 即可
  - 阅读理解、序列标注：需要更长 Prompt

  

- 引入多任务 Prompt 预训练

  

  - 缓解 Prompt 随机初始化带来的优化困难
  - 提升在复杂任务上的泛化能力

  

- 回归传统分类头设计

  

  - 不再依赖 Label Word Verbalizer
  - 使用标准 Classification Head
  - 显著提升在序列标注任务中的适用性

  





------





## **八、实验结果总结**





实验结果显示：



- 在 330M–10B 不同规模模型上，P-Tuning v2 均可达到与全参数微调相当的性能
- 在 RTE、NER、QA、SRL 等复杂序列任务上，P-Tuning v2 明显优于 Prompt Tuning 与 P-Tuning v1
- Prompt Length 对性能影响显著，且随任务复杂度提升而增加





这些结果表明，P-Tuning v2 在规模与任务维度上均具备良好的通用性。



------





## **九、方法定位总结**





从方法演进角度看：



- **P-Tuning**：对 Prompt Tuning 的连续化与可微改进
- **P-Tuning v2**：将 Prefix Tuning 的深度思想引入 NLU 场景，并进行工程化优化





P-Tuning v2 在多个模型规模与任务类型中展现出与全参数微调相媲美的性能，可作为参数高效微调方法的重要基线之一。



------



