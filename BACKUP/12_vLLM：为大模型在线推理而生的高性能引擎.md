# [vLLM：为大模型在线推理而生的高性能引擎](https://github.com/lihe/MyGitBlog/issues/12)







在真实的线上大模型服务中，推理瓶颈往往并不来自模型算力，而是来自**显存管理与并发调度**。

vLLM 正是为解决这一问题而诞生的推理引擎。



它并不是对 Transformer 结构的改写，而是一次**面向系统层的工程重构**。



------





## **一、vLLM 要解决的核心问题**





在大规模 LLM 在线服务中，推理面临几个根本性挑战：



- **KV Cache 占用巨大**

  KV Cache 随上下文长度线性增长，显存压力极高

- **请求长度高度不一致**

  长请求会拖垮短请求，batch 极易被打散

- **显存碎片严重**

  连续分配 KV Cache 容易导致 OOM

- **并发吞吐与 TTFT 难以兼顾**

  首 token 延迟（TTFT）与整体吞吐存在冲突





传统 transformers.generate() 的设计假设是：



> 单请求或静态 batch



这在实验环境中可行，但在高并发线上服务中几乎不可用。



vLLM 的目标很明确：



> **在显存受限条件下，实现高并发、低延迟、高吞吐的大模型推理。**



------





## **二、vLLM 的核心技术：Paged KV Cache（PagedAttention）**







### **1. 问题本质：KV Cache 的连续内存假设**





在标准自回归推理中：



- 每个请求需要一块**逻辑连续**的 KV Cache
- KV Cache 随 token 生成不断增长
- GPU 显存需要提前或动态分配大块连续空间





结果是：



- 内存碎片严重
- 长请求极易导致 OOM
- 并发请求之间难以共存





------





### **2. vLLM 的核心思想：像操作系统一样管理显存**





vLLM 引入 **PagedAttention**，其设计理念直接来自操作系统的**虚拟内存机制**：



- 将 KV Cache 划分为固定大小的 **page（block）**
- 每个 page 存储固定数量的 token
- 请求的 KV Cache 在逻辑上是连续的
- 在物理显存中是**非连续的**





通过 **page table** 建立逻辑序列到物理 page 的映射关系。



------





### **3. PagedAttention 带来的直接收益**





- 不再需要连续显存分配
- 显存碎片大幅减少
- KV Cache 可动态分配、回收、复用
- 长短请求可以安全共存
- 并发请求规模显著提升





> **vLLM 的吞吐优势，核心来自 Paged KV Cache，而不是单纯的算子加速。**



------





## **三、第二个关键技术：Continuous Batching（连续批处理）**


<img width="888" height="490" alt="Image" src="https://github.com/user-attachments/assets/31720f3c-cdaa-49c4-a549-f5030502ab8e" />




### **1. 传统 Static Batching 的问题**





传统 batching 方式要求：



- 请求同时到达
- 序列长度对齐
- batch 在整个生成过程中保持不变





现实中这几乎不成立：



- 有的请求很快结束
- 有的请求生成很长
- GPU 计算资源被频繁浪费





------





### **2. vLLM 的解法：每一步动态重组 batch**





vLLM 使用 **Continuous Batching**：



- 每一个 decoding step 都重新构建 batch
- 新请求可以随时加入
- 已完成请求立即移除
- batch 始终由“当前活跃请求”组成





配合 KV Cache：



- Q 的计算可批量执行
- K / V 从各自的 page cache 中读取





------





### **3. Continuous Batching 的效果**





- GPU 利用率显著提高
- TTFT（首 token 延迟）降低
- 并发吞吐最大化
- 不再被最慢请求拖累





------





## **四、Paged KV Cache × Continuous Batching 的协同效应**





这两项技术是**强耦合设计**：



- Paged KV Cache：

  

  - 解决“显存怎么分配”

  

- Continuous Batching：

  

  - 解决“算力怎么用满”

  





只有两者结合，才能真正支撑：



> **高并发 + 长上下文 + 低延迟**



这也是 vLLM 与传统推理框架的本质分界线。



------





## **五、vLLM 的完整推理工作流程**





一个请求在 vLLM 中的生命周期如下：



1. 请求到达 → 调度器接管

2. 为该请求分配 KV Cache pages

3. 请求进入活跃池

4. 每一个 decoding step：

   

   - 从活跃请求中动态组 batch
   - 从各自 page 中读取 KV Cache
   - 计算下一个 token

   

5. 请求结束：

   

   - 释放其 KV Cache pages
   - 立刻调度新的请求进入

   





整个过程没有“空 batch”，也没有“显存僵尸”。



------





## **六、vLLM 的其他关键特性（工程加分点）**





- **支持 MQA / GQA**

  

  - 从模型结构层减少 KV Cache 体积

  

- **FlashAttention 集成**

  

  - 优化 attention 计算本身

  

- **Tensor Parallelism**

  

  - 将模型权重切分到多 GPU

  

- **流式输出**

  

  - 支持 token 级别返回，降低感知延迟

  

- **OpenAI-compatible API**

  

  - 可直接替换现有服务

  

- **HuggingFace 模型兼容**

  

  - 无需重新训练或改模型结构

  





------





## **七、vLLM vs HuggingFace Transformers**



| **维度**      | **HF Transformers** | **vLLM** |
| ------------- | ------------------- | -------- |
| KV Cache 管理 | 连续分配            | 分页管理 |
| Batching      | 静态                | 连续动态 |
| 并发能力      | 低                  | 高       |
| TTFT          | 高                  | 低       |
| 显存利用率    | 差                  | 优       |
| 适用场景      | 实验 / 单请求       | 线上服务 |

一句话总结：



> **HF 适合研究，vLLM 适合生产。**



------





## **八、vLLM 的代价与边界**





vLLM 并不是万能的：



- 专注于推理，不支持训练
- 自定义模型 / 算子需要适配
- 极端超长上下文仍受显存限制（需 offload）





但在**线上 LLM 服务场景**，它几乎是当前的事实标准。



------





## **总结**





vLLM 是一个为大模型在线推理而设计的高性能引擎，其核心创新在于 **Paged KV Cache** 和 **Continuous Batching**。

它通过操作系统级的显存管理方式和动态调度机制，解决了传统 Transformer 推理在并发、显存碎片和吞吐上的根本瓶颈，显著提升了 GPU 利用率和系统整体性能。



**vLLM 的价值不在模型，而在系统。**

