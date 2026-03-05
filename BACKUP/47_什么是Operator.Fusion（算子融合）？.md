# [什么是Operator Fusion（算子融合）？](https://github.com/lihe/MyGitBlog/issues/47)





# **一、什么是算子融合（Operator Fusion）**





先看一个普通神经网络计算：

```
x → LayerNorm → GELU → Add → output
```

如果不做融合：



GPU执行流程是：

```
Kernel1: LayerNorm
Kernel2: GELU
Kernel3: Add
```

流程：

```
显存 → Kernel1 → 写回显存
显存 → Kernel2 → 写回显存
显存 → Kernel3 → 写回显存
```

问题是：

```
显存读写非常慢
```

GPU计算其实很快，但：



> **显存访问是最大瓶颈**



------





# **二、算子融合的核心思想**





算子融合：

```
LayerNorm + GELU + Add
        ↓
一个 CUDA kernel
```

执行：

```
读取一次显存
GPU计算
写回一次显存
```

从：

```
3次 kernel
6次显存访问
```

变成：

```
1次 kernel
2次显存访问
```

性能提升非常明显。



------





# **三、为什么 GPU 需要 Fusion**





GPU的执行模型：

```
CPU
 ↓
Launch CUDA Kernel
 ↓
GPU执行
```

每启动一次 kernel 都有：

```
kernel launch overhead
```

大约：

```
5–20 μs
```

如果模型有：

```
1000个小算子
```

启动时间就很大。



算子融合减少：

```
kernel launch 次数
```



------





# **四、GPU最大瓶颈：Memory Bound**





深度学习算子有两类：





### **1 计算密集**





例如：

```
Matrix Multiply
```

特点：

```
FLOPS 很多
```



------





### **2 内存密集**





例如：

```
ReLU
LayerNorm
Add
Softmax
```

特点：

```
计算少
读写多
```

所以：



> **这些算子非常适合 fusion**



------





# **五、三种常见融合方式**





图中提到三种：

```
垂直融合
水平融合
混合融合
```

我给你解释清楚。



------





# **1 垂直融合（Vertical Fusion）**





垂直融合 = **算子流水线合并**



例如：

```
LayerNorm
   ↓
GELU
   ↓
Add
```

融合成：

```
FusedLayerNormGELUAdd
```

流程：

```
load x
compute LayerNorm
compute GELU
compute Add
store result
```

只访问显存一次。



------





# **Transformer中的例子**





非常经典：

```
Bias + GELU
```

代码原来是：

```
x = linear(x)
x = x + bias
x = gelu(x)
```

融合后：

```
FusedBiasGELU
```



------





# **2 水平融合（Horizontal Fusion）**





水平融合 = **多个相同算子合并**



例如：



Transformer中：

```
Q = XWq
K = XWk
V = XWv
```

原来是：

```
MatMul1
MatMul2
MatMul3
```

融合为：

```
MatMul([Wq,Wk,Wv])
```

代码：

```
W = torch.cat([Wq, Wk, Wv], dim=1)
QKV = X @ W
```

一次矩阵乘法。



------





# **3 混合融合（Hybrid Fusion）**





典型：

```
Attention
```

FlashAttention 就是：

```
QK^T
Softmax
V
```

全部融合。



传统流程：

```
QK^T
↓
store
↓
Softmax
↓
store
↓
Matmul
```

FlashAttention：

```
Tile计算
不存中间矩阵
```

直接：

```
Attention output
```

显存访问减少非常多。



------





# **六、Transformer中的 Fusion**





Transformer有大量可融合算子：





### **1 LayerNorm Fusion**



```
LayerNorm + Residual
```

融合：

```
FusedLayerNormResidual
```



------





### **2 Bias + Activation**



```
Bias + GELU
Bias + ReLU
```



------





### **3 Attention Fusion**



```
QKV projection
Attention
Softmax
Dropout
Matmul
```

FlashAttention 就是融合。



------





# **七、图中 CUDA Kernel 示例解释**





代码：

```
__global__ void fused_layernorm_residual(
 float* output,
 const float* input,
 const float* residual)
```

做的事情：





### **Step1 残差连接**



```
val = input + residual
```



------





### **Step2 计算均值**



```
mean = block_reduce_mean(val)
```



------





### **Step3 计算方差**



```
var = block_reduce_var(val)
```



------





### **Step4 LayerNorm**



```
output = (val - mean)/sqrt(var+eps)
```

整个过程：

```
一个kernel完成
```

而不是：

```
Add kernel
LayerNorm kernel
```



------





# **八、真实框架如何做 Fusion**





很多框架自动做：





### **TensorRT**





自动：

```
Conv + BN + ReLU
```

融合。



------





### **PyTorch 2.0**





使用：

```
TorchInductor
```

自动 fusion。



------





### **XLA**





TensorFlow TPU 编译器：

```
XLA
```

做 graph fusion。



------





### **TVM**





深度学习编译器：

```
TVM
```

自动优化。



------





# **九、Fusion 的典型案例**







### **CNN**



```
Conv + BN + ReLU
```

融合为：

```
FusedConvBNReLU
```



------





### **Transformer**



```
Linear + Bias + GELU
```



------





### **Attention**



```
FlashAttention
```



------





# **十、Fusion带来的性能提升**





典型提升：

```
1.5x - 3x
```

FlashAttention：

```
2x - 4x
```

推理系统：

```
TensorRT
vLLM
DeepSpeed
```

都依赖 Fusion。



------





# **十一、什么时候不能融合**





Fusion不是万能的。



限制：





### **1 依赖关系复杂**





例如：

```
控制流
```



------





### **2 内存太大**





可能超出：

```
shared memory
```



------





### **3 数值稳定性**





例如：

```
Softmax
```

需要特殊处理。



------





# **十二、面试标准回答**





如果面试问：



**什么是 Operator Fusion？**



可以回答：



> Operator Fusion 是一种深度学习系统优化技术，通过将多个连续算子合并为一个 CUDA kernel 执行，从而减少显存读写和 kernel 启动开销。常见融合包括垂直融合（如 LayerNorm + GELU + Add）、水平融合（如 QKV projection 合并矩阵乘法）和混合融合（如 FlashAttention）。这种技术可以显著减少 memory bandwidth 开销，提高 GPU 利用率，是 TensorRT、TVM、XLA 等推理框架的重要优化手段。



------



如果你愿意，我可以继续给你讲一个 **非常关键但很多人不会的内容**：



**为什么 FlashAttention 能比普通 Attention 快 2–4 倍？**



里面涉及：

```
IO complexity
GPU memory hierarchy
tiling algorithm
```

这是 **大模型系统面试的高频题**。