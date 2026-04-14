[Remote Sensing Image Change Detection With Transformers | IEEE Journals & Magazine | IEEE Xplore](https://ieeexplore.ieee.org/document/9491802)

## 背景

拟解决的核心问题：如何在双时相高分辨率遥感图像中高效建模全时空语义关系，从而更准确地识别真正的变化并排除无关变化

更具体地，作者认为一个好的 CD 模型需要做到两件事：

- 理解高层语义，如空地变成建筑、植被变成道路，而不是只看像素值差异
- 排除无关变化，如光照、阴影和季节发生变化使得同类目标外观不同，但并不是任务真正关心的变化

作者认为现有方法主要有两类不足：

- 纯 CNN 方法擅长局部特征提取，但不擅长直接建模。也就是说很难处理**时空中的长程依赖**
- 非局部/Self-Attention 方法能够建模全局关系，但是要在大量像素之间计算两两关系，计算复杂度高。在高分辨率遥感图像中这个问题会更加严重



## 方法

### 核心思想

作者认为：

- 感兴趣的变化可以由少量高层语义概念表示。也就是说图像中真正重要的变化信息不一定要靠所有像素两两建模，可以先概括为少量 **语义 token**
- 不要直接在像素空间做全局注意力，计算量为 $O(N^2)$ 难以接受

因此作者提出了： **先把图像压缩成少量 token，再在 token 之间建模全局关系**

以此为基础，作者提出了 **BIT（Bitemporal Image Transformer，双时相图像 Transformer）** 变化检测框架

### 实现

BIT 有三个主要组成部分

#### 1. Siamese Semantic Tokenizer

孪生语义 tokenizer

作用：把每个时相的像素特征图压缩成少量语义 token

设每个时相提取到 $L$ 个 token，则输出为 $T_1, T_2 \in \mathbb{R}^{L \times C}$

这里：

- $L$：token 个数，也叫 vocabulary size，论文最终实验里取 $L = 4$
- 每个 token 为$C$ 维向量

具体做法：

- 用 CNN 提取每个时相的特征图 $X_1, X_2 \in \mathbb{R}^{HW \times C}$

- 对每个特征图通过一个逐点卷积把每个像素映射到 $L$ 个语义组上:
  $$
  \phi(X_i;W)
  $$
  其中 $W\in \mathbb{R}^{C\times L}$，可以理解为对每个像素的 $C$ 维特征，预测在 $L$ 给语义概念上的响应，于是得到一个形状为  $\mathbb{R}^{HW \times L}$ 的张量

- 计算空间注意力图

​	接着对每个语义组在空间维度 $HW$ 上做 softmax，得到注意力图：
$$
A_i = \sigma(\phi(X_i;W)) \\
A_i \in \mathbb{R}^{HW \times L}
$$
​	注意，这里的 softmax 不是对通道做，而是对每个 token 对应的整张空间图归一化

​	对于第 $l$ 个 token 有
$$
\sum_{p=1}^{HW} A_i(p,l) = 1
$$
​	可以理解为第 $l$ 个 token 会从整张图中挑出自己最关注的部分

- 最后用这些注意力图对原特征图做加权平均，得到 token 集
  $$
  T_i = A_i^TX_i \in \mathbb{R} ^ {L \times C}
  $$
  理解为每个 token 是对整张图中特定语义概念区域的加权汇聚结果

#### 2. Transformer Encoder

作用是在 token 空间里建模双时相图像的全局时空上下文关系

Transformer 编码器由 $N_E$ 层多头自注意力和多层感知机组成

前面得到两个时相的 token 集 $T_1, T_2 \in \mathbb{R}^{L \times C}$ ，拼接为 $T = Concat (T_1, T_2) \in \mathbb{R}^{2L \times C}$

然后加位置编码 $T^{(0)} = T + W_{PE}$，本文使用可学习位置编码

然后进入多层 encoder

论文用多头自注意力，输入为 $T^{(l-1)}$：
$$
\mathrm{MSA}(T^{(l-1)})
=
\mathrm{Concat}(head_1,\dots,head_h)W_O
$$
其中第 $j$ 个头是：

$$
head_j
=
\mathrm{Att}(T^{(l-1)}W_q^j,\; T^{(l-1)}W_k^j,\; T^{(l-1)}W_v^j) \\
W_q^j, W_k^j, W_v^j \in \mathbb{R}^{C \times d}
$$

实验中设置 $h=8$，每头维度 $d=8$



每层 encoder 还有一个前馈网络 MLP：
$$
\mathrm{MLP}(T^{(l-1)})
=
\mathrm{GELU}(T^{(l-1)}W_1)W_2
$$
其中：
$$
W_1 \in \mathbb{R}^{C \times 2C},\quad
W_2 \in \mathbb{R}^{2C \times C}
$$
也就是说：输入维度是 $C$，中间扩大到 $2C$，再投回 $C$



encoder 采用 PreNorm

虽然正文没把完整残差式子展开，但按标准写法可写成：

$$
\hat T^{(l)}
=
T^{(l-1)}
+
\mathrm{MSA}(\mathrm{LN}(T^{(l-1)})) 
\\
T^{(l)}=T^{(l)}+MLP(\mathrm{LN}(T^{(l)}))
$$

#### 3. Transformer Decoder

前面我们已经得到了

- 像素特征图 $X_i \in \mathbb{R}^{HW \times C}$
- 富含上下文 （Context-Rich）的 token $T_i^{new} \in \mathbb{R}^{L \times C} $

现在需要把 token 中的全局语义信息送回像素空间



Decoder 中使用交叉注意力，query 来自特征图，token 作为 key / value

总体结构类似于 Encoder

!!! note "复杂度分析"
	直接在像素空间做自注意力的复杂度是 $O((HW)^2)$ ，但是这里的交叉注意力里 query 数量为 $HW$，key / value 数量是 $L$，复杂度为 $O(HWL)$，由于  $L≪HW$ ，所以复杂度低很多

最终的输出是细化后的特征图 $X_i^{new} \in \mathbb{R}^{HW \times C}$

#### 4. Pediction Head

现在我们得到了两个增强特征图 $X_1^∗,X_2^* \in R^{H_0×W_0×C}$

首先先计算特征差分图
$$
D = |X_1^* - X_2^*|
$$
然后经过分类器 $g$ 和逐像素 softmax
$$
P = \sigma(g(D))
$$
其中分类器 $g: R^{H_0×W_0×C} \rightarrow R^{H_0×W_0×2}$ 把 $D$ 映射到两通道概率图 

#### 5. 损失函数

训练阶段通过最小化交叉熵损失来优化网络参数
$$
L = \frac{1}{H_0W_0} \sum_{h=1}^{H_0} \sum_{w=1}^{W_0} l(P_{hw},Y_{hw})
$$
$$
l(P_{hw},y) = -\log(P_{hwy})
$$

## 实验

效果超越 SOTA

### 消融实验

**消融编码器**

效果明显下降。使用 Non-local 自注意力层替换 BIT 效果不如 BIT，作者认为可能是因为 BIT 在基于 token 的空间中学习上下文，该空间更加紧凑，信息密度更高

**消融 tokenizer**

此时的模型可视为使用稠密 token，即直接从 CNN 骨干中提出的特征序列，效果明显下降。作者认为 tokenizer 通过空间池化稠密特征来聚合语义信息得到紧凑的概念 token

**消融解码器**

作者使用了一件简单模块代替它，用于融合来自编码器的 token 和来自 CNN 的原始特征，具体地说，将每个 token 的空间维展开，然后将 $L$ 个扩展 token 和 $X_i$ 相加，生成更新后的特征。效果明显下降

**位置嵌入的影响**

解码器中的 query 中添加位置嵌入没有带来明显提升，原因可能是编码器中的 key （即 token）本身高度抽象，并不包含空间结构。因此在最终模型中只在编码器中加入位置嵌入而不在解码器中

### 参数分析

**token 长度**

实验表明 token 长度从 32 降至 4 时效果明显提升，表明一个紧凑的 token 足以表示感兴趣的变化的语义概念

进一步把 $L$ 从 4 降至 2 时效果略有下降，因为此时模型可能会丢失一些与变化概念有关的有用信息

**Transformer 深度**

实验表明增加编码器深度并未带来明显提升，而性能与解码器深度大致呈正相关，最佳结果出现在 8

可能是因为：一层编码器就足以学习双时相 token 之间的关系。在解码器的每一层中图像特征都会结合 token 被进一步细化

## 讨论

本文提出了一种高效方法，用于在高分辨率遥感图像中执行变化检测。BIT 将图像表示为少量 token 向量，这种高信息密度的表示可能提升了训练效率。

BIT 也可以看作是一种高效的基于注意力的方式，用于扩大模型感受野，从而增强变化识别所需的特征表示能力。
