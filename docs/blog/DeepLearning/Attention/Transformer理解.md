# Transformer 理解

## 架构理解

首先 Transformer 的架构如下

![transformer](img/transformer.svg)

## 编码器：多头注意力

视为“所有 token 互相询问：“现在这句话里，谁对我最重要”

- 每个 token 有一个询问 (Query) ：我现在想知道什么
- 同时对外展示“我能回答什么”（键/key）
- 同时有着“如果来找我，我能提供的内容”（值/Value）

对于每个 token，可视作对其他所有 token 的 key 进行打分，得到一排关注度，再把所有 token  的关注度加权，可理解为根据上下文调整自身词义



### 编码器：逐位前馈网络 FFN

每个 token 各自分别过一遍 MLP 以消化自身信息，引入非线性



### 解码器：掩蔽多头注意力

屏蔽未来的信息，只保留已知信息



### 解码器：多头注意力

解码器每写一个词，回头看原文此时最该关注源句的哪些位置



### 解码器：逐位前馈网络 FFN

和编码器 FFN 类似，对每个向量本地加工，进行预测



### 解码器顶部：全连接层

把结果 softmax 成概率分布



## 复杂度理解

深入讨论一下 $O(n^2)$ 的复杂度是哪来的

以标准 Transformer 设以下参数：

- 序列长度：$n$
- 隐藏维度：$d$
- 头数：$h$
- 每个头的维度：$d_k = d/h$
- FFN 层中间维度：$d_{ffn}$，一般设为 $4d$

### 时间复杂度

#### 自注意力复杂度

首先是线性映射生成 $Q,K,V$

输入 $X \in \mathbb{R}^{n \times d}$

有 $Q = XW_Q, K = XW_K,V = XW_V$

矩阵运算复杂度 $O(nd^2)$

然后是计算注意力分数 $QK^T$，每个头 $Q,K \in \mathbb{R}^{n \times d_k}$，复杂度为 $O(n^2d_k)$，一共 $h$ 个头，复杂度 $O(hn^2d_k) = O(n^2 d)$

然后对分数 softmax，复杂度 $O(n^2)$，忽略

最后权重乘以 $V$，复杂度 $O(n^2d)$

合并为 $O(n^2d + nd^2)$

#### FFN 复杂度

$$
FFN(x)  = W_2 \sigma(W_1x+b_1)+b_2
$$

第一层：$d \rightarrow d_{ffn}$，复杂度 $O(ndd_{ffn})$

第二层：$d_{ffn} \rightarrow d$，复杂度 $O(ndd_{ffn})$

合并为 $O(ndd_{ffn}) = O(nd^2)$

#### 总结

合并以上结论可得 Transformer 的时间复杂度为 $O(nd^2 + n^2d)$

### 空间复杂度

对于 $Q,K,V$ 都是 $n \times d$，复杂度 $O(nd)$

注意力矩阵为 $h$ 个 $n \times n$ 的矩阵，通常将 $h$ 视为常数，复杂度 $O(n^2)$

对于 FFN 层，线性映射权重矩阵复杂度 $O(nd_{ffn})$

总空间复杂度为 $O(n^2 + nd)$，更常见地强调为 $O(n^2)$
