# 基于深度学习的网络流量分类及异常检测研究

# ideas

![1581675598504](C:\Users\liuxuechao\AppData\Roaming\Typora\typora-user-images\1581675598504.png)

不同特征检测

![1581760674285](C:\Users\liuxuechao\AppData\Roaming\Typora\typora-user-images\1581760674285.png)

![](C:\Users\liuxuechao\AppData\Roaming\Typora\typora-user-images\1581760782327.png)

![1581832471306](C:\Users\liuxuechao\AppData\Roaming\Typora\typora-user-images\1581832471306.png)

![1581832538121](C:\Users\liuxuechao\AppData\Roaming\Typora\typora-user-images\1581832538121.png)

![1581832634982](C:\Users\liuxuechao\AppData\Roaming\Typora\typora-user-images\1581832634982.png)

![1581833789434](C:\Users\liuxuechao\AppData\Roaming\Typora\typora-user-images\1581833789434.png)

![1581854018936](C:\Users\liuxuechao\AppData\Roaming\Typora\typora-user-images\1581854018936.png)

![1581854053853](C:\Users\liuxuechao\AppData\Roaming\Typora\typora-user-images\1581854053853.png)

![1581855161676](C:\Users\liuxuechao\AppData\Roaming\Typora\typora-user-images\1581855161676.png)

# 2.15

## 分类方法

### 基于端口的分类方法

### 基于深层包检测的分类方法

指纹匹配

### 基于统计的分类方法

机器学习

#### 流量特征

网络流特征 数据包特征

![1581760604430](C:\Users\liuxuechao\AppData\Roaming\Typora\typora-user-images\1581760604430.png)

无监督检测误警率较高

### 基于行为的分类方法

根据主机通信的行为信息，例如主机与其他多少主机通信，分别采用哪种行为和端口，同样使用机器学习方法

## 恶意流量分类

### 基于指纹的恶意流量分类

### 基于异常/行为的恶意流量分类方法

## 加密流量分类

### 基于载荷的分类方法

不同的加密方法都有固定的协议格式

### 基于特征的分类方法

## 网络流量异常检测方法

![1581766566294](C:\Users\liuxuechao\AppData\Roaming\Typora\typora-user-images\1581766566294.png)

# 2.16

## 基于聚类的检测方法

常规聚类 协同聚类

![1581778717911](C:\Users\liuxuechao\AppData\Roaming\Typora\typora-user-images\1581778717911.png)

## *反向传播算法*

![1581830484379](C:\Users\liuxuechao\AppData\Roaming\Typora\typora-user-images\1581830484379.png)

训练时每次更新权重时偏导数求解问题

反向传播算法

![1581831671243](C:\Users\liuxuechao\AppData\Roaming\Typora\typora-user-images\1581831671243.png)

![1581831711836](C:\Users\liuxuechao\AppData\Roaming\Typora\typora-user-images\1581831711836.png)



## CNN端到端加密流量分析

![1581756249448](C:\Users\liuxuechao\AppData\Roaming\Typora\typora-user-images\1581756249448.png)

## 基于表征学习的分类

![1581833774638](C:\Users\liuxuechao\AppData\Roaming\Typora\typora-user-images\1581833774638.png)

#### 对流量数据进行单元划分

包

原始流量

flow

session

选取flow或者session的前n个字节

#### 协议层次

#### 网络流量处理

可视化

![1581853383138](C:\Users\liuxuechao\AppData\Roaming\Typora\typora-user-images\1581853383138.png)

