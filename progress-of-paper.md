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

### RNN(递归神经网络)



