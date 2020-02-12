##  advanced persistant threat 

 advanced persistant threat 高级持续性攻击，提高获取信息的权限 

## botnet

| IRC协议  |                    |
| -------- | :----------------: |
| HTTP协议 | 周期性访问不易侦察 |
| P2P协议  |      去中心化      |



## 基于统计分析的异常检测技术

捕获流量活动并且构建合法行为轮廓

### 均值/标准差方法

### 多变量

### 马尔可夫过程

## 基于数据挖掘的异常检测技术

提取其中潜在的事先不明确的潜在信息

### 关联分析

描述某些属性同时出现的规律，apriori算法

### 序列分析

挖掘基于时间的顺序事件

### 聚类分析

基于K均值的算法分析方法（eg)

botminer

### 分类分析

#### 二分类任务

#### 多分类任务

异常行为分类为：

###### dos攻击

使得资源紧张，eg pingofdeath

###### u2r攻击

to be super user eg.buffer overflow attacks

r2l攻击

guessing password(社会工程学等)

###### probe攻击

端口扫描探测

## 基于特征工程的检测技术

### 特征降维

降低训练时间，测试时间，提高分类效率，准确率

### 特征选择

降低特征空间

## 基于机器学习的网络异常检测技术

### 贝叶斯网络

有向无环图描述关系

### 遗传算法

模拟遗传变异选择的自适应算法

### 支持向量机	

找到最佳超平面，将样本分开eg恶意邮件分离

### K最近邻

### 决策树

### 模糊逻辑

### 人工神经网络

由具有适应性的简单单元组成的广泛互联的神经网络

### 深度学习

## 深度学习存在问题

![1580466755987](C:\Users\liuxuechao\AppData\Roaming\Typora\typora-user-images\1580466755987.png)

# 2.1学习内容

## 数据集

### 网络流量捕捉

问题：数据量过少

### 构造

问题：数据质量存在问题，隐私问题

但也有可用的质量较高的数据集

## 评价分类模型检测性能

### 混淆矩阵

![1580481635337](C:\Users\liuxuechao\AppData\Roaming\Typora\typora-user-images\1580481635337.png)

#### 准确率

正常和攻击行为被检测概率

(TP+TN)/ALL

#### 精确率

TP/（TP+FP)

#### 召回率

攻击行为被正确检测概率

TP/(TP+FN)

#### 误报率

正常行为被错误的检测为异常攻击的比率

FP/(TN+FP)

## 循环神经网络

结合上下文判断，记忆了T之前所有信息并作用于以后时刻，理论上学习任何时长



<img src="C:\Users\liuxuechao\AppData\Roaming\Typora\typora-user-images\1580483706137.png" alt="1580483706137" style="zoom:50%;" />





### 基于全连接循环神经网络的入侵检测系统（RNN-IDS)

#### 数值化

#### 归一化（0，1）

#### 前向传播

计算每一个样本输出并预测标签

#### 权值微调（根据预测与实际差距）

#### 检测分类（测试集）

<img src="C:\Users\liuxuechao\AppData\Roaming\Typora\typora-user-images\1580484477553.png" alt="1580484477553" style="zoom:50%;" />

#### 前向传播算法

![1580484814632](C:\Users\liuxuechao\AppData\Roaming\Typora\typora-user-images\1580484814632.png)

#### 权值微调算法

### 训练结果

在二分类任务下表现优秀，多分类下也较好，在无GPU加速的情况下，训练时长较长

## 基于生成式对抗网络的入侵检测技术/GAN

 用于生成/改造/变化等 

###  generator 

 训练目标：接近真实 随机噪声经过反卷积（生成器） 

### discriminator

 训练目标：提高识别能力（卷积神经网络CNN） 

###   D(G(z))=0.5 

D(G(z))=0.5 

### 训练过程

固定其中一个模型，修改另一个模型的参数，使得固定模型的错误最大化，博弈过程

### 有监督学习的生成式对抗网络的入侵检测框架（ID-GAN)

<img src="C:\Users\liuxuechao\AppData\Roaming\Typora\typora-user-images\1580526532576.png" alt="1580526532576" style="zoom: 50%;" />

#### 训练过程

由于参数较多，模型可选定几个影响较大的参数调整，得到局部最优

##### 先验训练

利用原有标签集对模型进行训练。防止加入生成集之后产生不可控

### 结论

先验训练有必要，不可混入过多生成样本防止系统检测重心偏差

## 机器学习

### 有监督学习

任务为分类，根据标签

### 无监督学习（大数据反欺诈）

聚类，数据挖掘，用于在大量无标签数据中发现些什么

选择合适的分类标准对结果很重要

# 2.5学习笔记

## 僵尸网络检测技术

### LSTM

把信息有效整合和筛选，具备了记住长期信息的能力

### 基于LSTM的僵尸网络检测技术

数值化

归一化

提取特征

![1580873046840](C:\Users\liuxuechao\AppData\Roaming\Typora\typora-user-images\1580873046840.png)

![1580873066587](C:\Users\liuxuechao\AppData\Roaming\Typora\typora-user-images\1580873066587.png)

### 基于生成式对抗网络的僵尸网络监测技术

### bot-GAN

更加注重检测模型而不是生成模型

#### 生成式对抗网络变体 

![1580873556170](C:\Users\liuxuechao\AppData\Roaming\Typora\typora-user-images\1580873556170.png)



#### bot-gan模型

![1580873690463](C:\Users\liuxuechao\AppData\Roaming\Typora\typora-user-images\1580873690463.png)

##### 训练

先验训练分类

## 总结发展

1.云计算的结合，高性能计算环境中实时监测能力

2.调参

# 2.7

### RvNN递归神经网络

当一个图像被分成不同的感兴趣的部分时，一个句子被分成几个词。RvNN计算可能的一对的得分来合并它们并构建一个语法树。对于每对单位，RvNN计算合并的合理性得分。得分最高的一对组合成一个组合向量。每次合并后，RvNN将生成（1）多个单元的更大区域

![1581067419988](C:\Users\liuxuechao\AppData\Roaming\Typora\typora-user-images\1581067419988.png)

![1581067841315](C:\Users\liuxuechao\AppData\Roaming\Typora\typora-user-images\1581067841315.png)

LSTM解决了前序过多造成的梯度消失

![1581067855913](C:\Users\liuxuechao\AppData\Roaming\Typora\typora-user-images\1581067855913.png)

attention 机制

![1581068012669](C:\Users\liuxuechao\AppData\Roaming\Typora\typora-user-images\1581068012669.png)

# 2.8

### RNN(递归神经网络)

![1581130142465](C:\Users\liuxuechao\AppData\Roaming\Typora\typora-user-images\1581130142465.png)



One main issue of an RNN is its sensitivity to the vanishing and exploding gradients（梯度）

### CNN（卷积神经网络）

对一小块像素区域的处理，使得神经网络可以看到图形，而非一个点

![1581131719434](C:\Users\liuxuechao\AppData\Roaming\Typora\typora-user-images\1581131719434.png)

![1581131816482](C:\Users\liuxuechao\AppData\Roaming\Typora\typora-user-images\1581131816482.png)

#### 半监督学习/semisupervised learning

使用预训练网络作为固定特征提取器（特别是对于小的新数据集）或微调预训练模型的权重（特别是对于与原始数据集相似的大的新数据集）来完成。



### 分布式系统

#### 数据并行

对于数据并行，将模型复制到所有计算节点，并使用指定的数据子集训练每个模型。**一定时间后，权值更新需要在节点间同步。**相比之下，对于模型的并行性，所有的数据都是用一个模型处理的，其中每个节点负责模型中参数的部分估计。

但是有通信成本

一种更流行的数据并行方法使用SGD，称为基于更新的数据并行

##### SGD

在实践中，SGD方法是应用于深度学习的基本算法，它根据每次训练的梯度迭代调整参数

#### 模型并行

模型并行方法将训练步骤分割为多个gpu。在一个简单的模型并行策略中，每个GPU只计算模型的一个子集。

模型并行策略的同步损失和通信开销大于数据并行策略，因为模型并行策略中的每个节点必须在每个更新步骤上同步梯度和参数值。

#### 静态图与动态图

。动态计算意味着程序将按照我们编写命令的顺序进行执行。这种机制将使得调试更加容易，并且也使得我们将大脑中的想法转化为实际代码变得更加容易。而静态计算则意味着程序在编译执行时将先生成神经网络的结构，然后再执行相应操作。从理论上讲，静态计算这样的机制允许编译器进行更大程度的优化，但是这也意味着你所期望的程序与编译器实际执行之间存在着更多的代沟。这也意味着，代码中的错误将更加难以发现（比如，如果计算图的结构出现问题，你可能只有在代码执行到相应操作的时候才能发现它）。**尽管理论上而言，静态计算图比动态计算图具有更好的性能，但是在实践中我们经常发现并不是这样的。**

# 2.10

## 深度学习的应用

### natural language processing/NLP

（1）通过标记化将输入文本分解成单词，然后（2）这些单词以向量或n-grams的形式再现。

#### sentiment analysis

情感分析中的大多数数据集被标记为正或负，中性短语被主观性分类方法去除。

**数据集**  在研究情绪时，社交媒体是一种流行的数据来源

提出了一种递归神经张量网络（RNTN），它利用词向量和解析树来表示短语，用基于张量的合成函数捕捉元素之间的交互作用。这种递归方法在句子级分类时是有利的，因为语法通常显示**树状结构**。

### machine translation

### 释义识别

提出使用展开递归自动编码器（RAEs）来度量两个句子的相似性。利用句法树来扩展特征空间，它们同时测量词和短语的相似性。

### summarization

一个基于连续向量表示的句子模型。他们的模型为有意义的表示评估多个组合和组合。

### question answering

提出了一个多栏CNN方法，可以从几个方面分析问题，即选择哪种上下文，答案的潜在语义，以及如何形成答案。他们使用一种多任务方法，对问答对进行排序，同时还学习低级词汇语义的相关性和关联性。提出了一种不局限于任何一种语言的更通用的深度学习体系结构

一个高度可扩展的问答模型。他们解决大数据集问题的方法是避免文本的逻辑形式，只在问答元组上学习模型。

通过根据候选答案与问题之间的相互依赖程度对候选答案进行排序

###### 

# 2.12

### visual data processing/可视化数据处理

#### image classification

###### **过度拟合问题**

泛化能力：对未知数据集有很好的拟合结果

过度拟合：由于训练数量太大，对训练集外的数据不work

解决：数据集增强，创建假的数据集

控制模型复杂度

L1/L2正则化，降低模型复杂度，

<img src="C:\Users\liuxuechao\AppData\Roaming\Typora\typora-user-images\1581475036779.png" alt="1581475036779" style="zoom:25%;" />

dropout

在隐藏单元中增加噪声

early stopping

在验证集上发现如果测试误差上升，则停止训练

###### 梯度消失问题

由于反向传播训练法则/激活函数选择不好，靠近输入的层学习很慢，靠近输出的层学习效果较好

解决：预训练+微调

逐层训练，微调（应用不多）整体性缺失

relu激活函数

LSTM

#### Object Detection and Semantic Segmentation/目标检测和语义分割

#### video proccessing

同时包含时空信息

一种新的视频处理技术被称为递归卷积网络（RCNs）。它将CNNs应用于视频帧上进行视觉理解，然后将帧反馈给RNNs进行视频中的时间信息分析。

传统的二维CNN相比，三维CNN（C3D）在视频分析任务上表现出了更好的性能。它从视频输入中自动学习时空特征，同时对图像的外观和运动进行建模。



# 2.13

支持向量机？

线性分类

贝叶斯决策？

logisitc回归?

无监督学习

Kmeans？