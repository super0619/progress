在VS CODE里面编辑

复制到工作区所在的文件夹

![1584177572473](C:\Users\liuxuechao\AppData\Roaming\Typora\typora-user-images\1584177572473.png)

然后在tensorflow打开

![1584177622247](C:\Users\liuxuechao\AppData\Roaming\Typora\typora-user-images\1584177622247.png)

![1583488092371](C:\Users\liuxuechao\AppData\Roaming\Typora\typora-user-images\1583488092371.png)

a）序贯模型（Sequential):单输入单输出，一条路通到底，层与层之间只有相邻关系，没有跨层连接。这种模型编译速度快，操作也比较简单

b）函数式模型（Model）：多输入多输出，层与层之间任意连接。这种模型编译速度慢。

-----------------------------------------------------------------------------------------------------------------------------------------------------------输入向量是784维度的，第一个影藏层是1000个节点，init代表的是链接矩阵中的权值初始化

'''

init 初始化参数:

uniform(scale=0.05) :均匀分布，最常用的。Scale就是均匀分布的每个数据在-scale~scale之间。此处就是-0.05~0.05。scale默认值是0.05；

lecun_uniform:是在LeCun在98年发表的论文中基于uniform的一种方法。区别就是lecun_uniform的scale=sqrt(3/f_in)。f_in就是待初始化权值矩阵的行。

normal：正态分布（高斯分布）。

identity ：用于2维方阵，返回一个单位阵

orthogonal：用于2维方阵，返回一个正交矩阵。

zero：产生一个全0矩阵。

glorot_normal：基于normal分布，normal的默认 sigma^2=scale=0.05，而此处sigma^2=scale=sqrt(2 / (f_in+ f_out))，其中，f_in和f_out是待初始化矩阵的行和列。

glorot_uniform：基于uniform分布，uniform的默认scale=0.05，而此处scale=sqrt( 6 / (f_in +f_out)) ，其中，f_in和f_out是待初始化矩阵的行和列。

he_normal：基于normal分布，normal的默认 scale=0.05，而此处scale=sqrt(2 / f_in)，其中，f_in是待初始化矩阵的行。

he_uniform：基于uniform分布，uniform的默认scale=0.05，而此处scale=sqrt( 6 / f_in)，其中，f_in待初始化矩阵的行。

设定参数

lr表示学习速率，decay是学习速率的衰减系数(每个epoch衰减一次)，momentum表示动量项，Nesterov的值是False或者True，表示使不使用Nesterov momentum。

sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9,nesterov=True)

loss代表的是损失函数, optimizer代表的是优化方法, class_mode代表

使用交叉熵作为loss函数，就是熟知的log损失函数

model.compile(loss='categorical_crossentropy',optimizer=sgd, class_mode='categorical')

![1583501724678](C:\Users\liuxuechao\AppData\Roaming\Typora\typora-user-images\1583501724678.png)

开始训练，这里参数比较多。batch_size就是batch_size，nb_epoch就是最多迭代的次数， shuffle就是是否把数据随机打乱之后再进行训练

verbose是屏显模式，官方这么说的：verbose: 0 for no logging to stdout, 1 for progress bar logging, 2 for one log line per epoch.

就是说0是不屏显，1是显示一个进度条，2是每个epoch都显示一行数据

show_accuracy就是显示每次迭代后的正确率

validation_split就是拿出百分之多少用来做交叉验证

```python

from keras.models import Sequential  

from keras.layers.core import Dense, Dropout, Activation  

from keras.optimizers import SGD  

from keras.datasets import mnist  

import numpy 

'''

    第一步：选择模型

'''

model = Sequential()

'''

   第二步：构建网络层

'''

model.add(Dense(500,input_shape=(784,))) # 输入层，28*28=784  

model.add(Activation('tanh')) # 激活函数是tanh  

model.add(Dropout(0.5)) # 采用50%的dropout

 

model.add(Dense(500)) # 隐藏层节点500个  

model.add(Activation('tanh'))  

model.add(Dropout(0.5))

 

model.add(Dense(10)) # 输出结果是10个类别，所以维度是10  

model.add(Activation('softmax')) # 最后一层用softmax作为激活函数

 

'''

   第三步：编译

'''

sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True) # 优化函数，设定学习率（lr）等参数  

model.compile(loss='categorical_crossentropy', optimizer=sgd, class_mode='categorical') # 使用交叉熵作为loss函数

 

'''

   第四步：训练

   .fit的一些参数

   batch_size：对总的样本数进行分组，每组包含的样本数量

   epochs ：训练次数

   shuffle：是否把数据随机打乱之后再进行训练

   validation_split：拿出百分之多少用来做交叉验证

   verbose：屏显模式 0：不输出  1：输出进度  2：输出每次的训练结果

'''

(X_train, y_train), (X_test, y_test) = mnist.load_data() # 使用Keras自带的mnist工具读取数据（第一次需要联网）

# 由于mist的输入数据维度是(num, 28, 28)，这里需要把后面的维度直接拼起来变成784维  

X_train = X_train.reshape(X_train.shape[0], X_train.shape[1] * X_train.shape[2]) 

X_test = X_test.reshape(X_test.shape[0], X_test.shape[1] * X_test.shape[2])  

Y_train = (numpy.arange(10) == y_train[:, None]).astype(int) 

Y_test = (numpy.arange(10) == y_test[:, None]).astype(int)

 

model.fit(X_train,Y_train,batch_size=200,epochs=50,shuffle=True,verbose=0,validation_split=0.3)

model.evaluate(X_test, Y_test, batch_size=200, verbose=0)

 

'''

    第五步：输出

'''

print("test set")

scores = model.evaluate(X_test,Y_test,batch_size=200,verbose=0)

print("")

print("The test loss is %f" % scores)

result = model.predict(X_test,batch_size=200,verbose=0)

 

result_max = numpy.argmax(result, axis = 1)

test_max = numpy.argmax(Y_test, axis = 1)

 

result_bool = numpy.equal(result_max, test_max)

true_num = numpy.sum(result_bool)

print("")

print("The accuracy of the model is %f" % (true_num/len(result_bool)))

```

(https://blog.csdn.net/zjw642337320/article/details/81204560)

![1583549393942](C:\Users\liuxuechao\AppData\Roaming\Typora\typora-user-images\1583549393942.png)

# IDEAS

1.预训练 对于多节点和深网络比较有用（基本被淘汰）

2.cycleGAN



https://tensorflow.google.cn/guide/keras/rnn#bidirectional_rnns

3.![1583506381956](C:\Users\liuxuechao\AppData\Roaming\Typora\typora-user-images\1583506381956.png)

4.协作式

# RNN

### /*关于词嵌入：编码Word embeddings

#### One-hot encodings

将csv文件中的文字转化成数据

https://blog.csdn.net/mewbing/article/details/83653637?depth_1-utm_source=distribute.pc_relevant.none-task&utm_source=distribute.pc_relevant.none-task

![1583919116224](C:\Users\liuxuechao\AppData\Roaming\Typora\typora-user-images\1583919116224.png)

Q：转化为列之后会不会产生与其价值不相符的影响，

#### Encode each word with a unique number

#### Word embeddings（词嵌入）

 一种使用高效密集表示的方式，其中相似的词具有相似的编码。重要的是，我们不必手动指定此编码。嵌入是浮点值的密集向量（向量的长度是您指定的参数）。它们不是可手动指定嵌入值的值，而是可训练的参数（模型在训练过程中学习的权重，就像模型学习密集层的权重一样）。 

![1583486011613](C:\Users\liuxuechao\AppData\Roaming\Typora\typora-user-images\1583486011613.png)

![1583486045320](C:\Users\liuxuechao\AppData\Roaming\Typora\typora-user-images\1583486045320.png)

https://blog.csdn.net/weixin_43763859/article/details/101716743?depth_1-utm_source=distribute.pc_relevant.none-task&utm_source=distribute.pc_relevant.none-task （tfds.builder tfds.load)

![1583486964407](C:\Users\liuxuechao\AppData\Roaming\Typora\typora-user-images\1583486964407.png)

![1583504873083](C:\Users\liuxuechao\AppData\Roaming\Typora\typora-user-images\1583504873083.png)

![1583505002944](C:\Users\liuxuechao\AppData\Roaming\Typora\typora-user-images\1583505002944.png)

## */

RNN基本设定https://tensorflow.google.cn/guide/keras/rnn

```python
from __future__ import absolute_import, division, print_function, unicode_literals
import collections
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
```

![1583487948865](C:\Users\liuxuechao\AppData\Roaming\Typora\typora-user-images\1583487948865.png)

https://tensorflow.google.cn/api_docs/python/tf/keras/layers (layers模组)

![1583504701677](C:\Users\liuxuechao\AppData\Roaming\Typora\typora-user-images\1583504701677.png)

embedding层

参数：

![1583505084161](C:\Users\liuxuechao\AppData\Roaming\Typora\typora-user-images\1583505084161.png)

![1583505536863](C:\Users\liuxuechao\AppData\Roaming\Typora\typora-user-images\1583505536863.png)

dense层可以做加减与0比较，and or，以及乘法（https://blog.csdn.net/weixin_39862845/article/details/79275671）

LSTM（http://blog.sina.com.cn/s/blog_b9899d9f01031i5j.html）输出维度就是并行处理单元的数量

Keras中使用LSTM，只需指定LSTM 层的输出维度，其他所有参数（有很多）都使用 Keras 默认值。Keras 具有很好的默认值，无须手动调参，模型通常也能正常运行。

![1583507066920](C:\Users\liuxuechao\AppData\Roaming\Typora\typora-user-images\1583507066920.png)

##  基本配置 

测试代码

```python
>>> import tensorflow as tf
>>> a=3
>>> w=tf.Variable([[0.5,1.0]])
>>> x=tf.Variable([[2.0],[1.0]])
>>> y=tf.matmul(w,x)
>>> init_op=tf.global_variables_initializer()
>>> with tf.Session() as sess:
...     sess.run(init_op)
...     print (y.eval())
[[2.]]
```

 第一个简单测试用例通过（其中存在CPU support问题 忽略/优化） 

# CNN框架函数研究https://tensorflow.google.cn/tutorials/images/cnn

###### 模型建立

```python
from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf
from tensorflow.keras import datasets, layers, models
import matplotlib.pyplot as plt
```

```python
from __future__ import absolute_import, division, print_function, unicode_literals
```

如果某个版本中出现了某个新的功能特性，而且这个特性和当前版本中使用的不兼容，也就是它在该版本中不是语言标准，那么我如果想要使用的话就需要从**future**模块导入。

这样的做法的作用就是将新版本的特性引进当前版本中，也就是说我们可以在当前版本使用新版本的一些特性。

```python
from tensorflow.keras import datasets, layers, models
```

datasets 数据集函数，下载

layers TensorFlow 中的 layers 模块提供用于深度学习的更高层次封装的 API，利用它我们可以轻松地构建模型

https://cuiqingcai.com/5715.html

![1582539598648](C:\Users\liuxuechao\AppData\Roaming\Typora\typora-user-images\1582539598648.png)

```python
matplotlib.pyplot 
```

每一个pyplot函数都使一副图像做出些许改变，例如创建一幅图，在图中创建一个绘图区域，在绘图区域中添加一条线等等。在matplotlib.pyplot中，各种状态通过函数调用保存起来，以便于可以随时跟踪像当前图像和绘图区域这样的东西。

```python
model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
```

###### 编译训练模型

```
model.compile()
```

其中参数包括：优化函数

loss

metrics: 列表，包含评估模型在训练和测试时的性能的指标，典型用法是metrics=[‘accuracy’]。如果要在多输出模型中为不同的输出指定不同的指标，可向该参数传递一个字典，例如metrics={‘output_a’: ‘accuracy’}。评价函数,与损失函数类似,只不过评价函数的结果不会用于训练过程中,可以传递已有的评价函数名称,或者传递一个自定义的theano/tensorflow函数来使用

```python
model.fit()
```

https://blog.csdn.net/a1111h/article/details/82148497参数问题

###### 评估模型

```python
plt.plot()
```

https://blog.csdn.net/Chenzhoiku/article/details/62889490

plt库

# DCGAN



# 2.1学习笔记

### 层堆叠模型试验 tf.keras.Sequential 模型 

keras框架更加注重深度学习模型构建，提供了便利的API接口，简单堆积即可实现

#### tf1.0运行结果

```python

(base) C:\Users\liuxuechao>activate tensorflow

(tensorflow) C:\Users\liuxuechao>python
Python 3.5.6 |Anaconda, Inc.| (default, Aug 26 2018, 16:05:27) [MSC v.1900 64 bit (AMD64)] on win32
Type "help", "copyright", "credits" or "license" for more information.

//常见的神经网络都包含在keras.layer中，导入

>>> import tensorflow as tf
>>> from tensorflow.keras import layers
>>> print(tf.__version__)
1.10.0
>>> print(tf.keras.__version__)
2.1.6-tf

//最常见的模型类型是层的堆叠：tf.keras.Sequential 模型

>>> model = tf.keras.Sequential()
>>> model.add(layers.Dense(32, activation='relu'))
>>> model.add(layers.Dense(32, activation='relu'))
>>> model.add(layers.Dense(10, activation='softmax'))

//网络配置
//activation：设置层的激活函数。
//kernel_initializer 和 bias_initializer：创建层权重（核和偏差）的初始化方案。
//kernel_regularizer 和 bias_regularizer：应用层权重（核和偏差）的正则化方案，

>>> layers.Dense(32, activation='sigmoid')
<tensorflow.python.keras.layers.core.Dense object at 0x0000023FA25D3668>
>>> layers.Dense(32, activation=tf.sigmoid)
<tensorflow.python.keras.layers.core.Dense object at 0x0000023F9B708D68>
>>> layers.Dense(32, kernel_initializer='orthogonal')
<tensorflow.python.keras.layers.core.Dense object at 0x0000023FA25D3748>
>>> layers.Dense(32, kernel_initializer=tf.keras.initializers.glorot_normal)
<tensorflow.python.keras.layers.core.Dense object at 0x0000023F9B708D68>
>>> layers.Dense(32, kernel_regularizer=tf.keras.regularizers.l2(0.01))
<tensorflow.python.keras.layers.core.Dense object at 0x0000023FA25D3748>
>>> layers.Dense(32, kernel_regularizer=tf.keras.regularizers.l1(0.01))
<tensorflow.python.keras.layers.core.Dense object at 0x0000023FA25D3668>

//构建好模型后，通过调用 compile 方法配置该模型的学习流程

>>> model = tf.keras.Sequential()
>>> model.add(layers.Dense(32, activation='relu'))
>>> model.add(layers.Dense(32, activation='relu'))
>>> model.add(layers.Dense(10, activation='softmax'))
>>> model.compile(optimizer=tf.keras.optimizers.Adam(0.001),
...              loss=tf.keras.losses.categorical_crossentropy,
...              metrics=[tf.keras.metrics.categorical_accuracy])
>>> import numpy as np
>>> train_x = np.random.random((1000, 72))
>>> train_y = np.random.random((1000, 10))
>>> val_x = np.random.random((200, 72))
>>> val_y = np.random.random((200, 10))
>>> model.fit(train_x, train_y, epochs=10, batch_size=100,
...           validation_data=(val_x, val_y))
Train on 1000 samples, validate on 200 samples
Epoch 1/10
2020-02-01 15:45:07.152274: I tensorflow/core/platform/cpu_feature_guard.cc:141] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX AVX2
2020-02-01 15:45:07.162919: I tensorflow/core/common_runtime/process_util.cc:69] Creating new thread pool with default inter op setting: 8. Tune using inter_op_parallelism_threads for best performance.
1000/1000 [==============================] - 1s 559us/step - loss: 11.7426 - categorical_accuracy: 0.1130 - val_loss: 11.4300 - val_categorical_accuracy: 0.0850
Epoch 2/10
1000/1000 [==============================] - 0s 29us/step - loss: 11.5270 - categorical_accuracy: 0.1040 - val_loss: 11.3669 - val_categorical_accuracy: 0.1150
Epoch 3/10
1000/1000 [==============================] - 0s 34us/step - loss: 11.5023 - categorical_accuracy: 0.0980 - val_loss: 11.3590 - val_categorical_accuracy: 0.0950
Epoch 4/10
1000/1000 [==============================] - 0s 30us/step - loss: 11.4971 - categorical_accuracy: 0.1120 - val_loss: 11.3537 - val_categorical_accuracy: 0.0850
Epoch 5/10
1000/1000 [==============================] - 0s 30us/step - loss: 11.4933 - categorical_accuracy: 0.1020 - val_loss: 11.3500 - val_categorical_accuracy: 0.0850
Epoch 6/10
1000/1000 [==============================] - 0s 32us/step - loss: 11.4902 - categorical_accuracy: 0.1120 - val_loss: 11.3484 - val_categorical_accuracy: 0.1100
Epoch 7/10
1000/1000 [==============================] - 0s 32us/step - loss: 11.4881 - categorical_accuracy: 0.1190 - val_loss: 11.3482 - val_categorical_accuracy: 0.0850
Epoch 8/10
1000/1000 [==============================] - 0s 31us/step - loss: 11.4865 - categorical_accuracy: 0.1210 - val_loss: 11.3481 - val_categorical_accuracy: 0.0950
Epoch 9/10
1000/1000 [==============================] - 0s 31us/step - loss: 11.4851 - categorical_accuracy: 0.1150 - val_loss: 11.3475 - val_categorical_accuracy: 0.1050
Epoch 10/10
1000/1000 [==============================] - 0s 30us/step - loss: 11.4839 - categorical_accuracy: 0.1090 - val_loss: 11.3474 - val_categorical_accuracy: 0.1150
<tensorflow.python.keras.callbacks.History object at 0x0000023FA25D3B00>
>>> dataset = tf.data.Dataset.from_tensor_slices((train_x, train_y))
>>> dataset = dataset.batch(32)
>>> dataset = dataset.repeat()
>>> val_dataset = tf.data.Dataset.from_tensor_slices((val_x, val_y))
>>> val_dataset = val_dataset.batch(32)
>>> val_dataset = val_dataset.repeat()
>>> model.fit(dataset, epochs=10, steps_per_epoch=30,
...           validation_data=val_dataset, validation_steps=3)
Epoch 1/10
30/30 [==============================] - 0s 13ms/step - loss: 11.4983 - categorical_accuracy: 0.1167 - val_loss: 11.2838 - val_categorical_accuracy: 0.1458
Epoch 2/10
30/30 [==============================] - 0s 2ms/step - loss: 11.4647 - categorical_accuracy: 0.1083 - val_loss: 11.4630 - val_categorical_accuracy: 0.0833
Epoch 3/10
30/30 [==============================] - 0s 2ms/step - loss: 11.4719 - categorical_accuracy: 0.1135 - val_loss: 11.1501 - val_categorical_accuracy: 0.1458
Epoch 4/10
30/30 [==============================] - 0s 2ms/step - loss: 11.4502 - categorical_accuracy: 0.1260 - val_loss: 11.2492 - val_categorical_accuracy: 0.1146
Epoch 5/10
30/30 [==============================] - 0s 2ms/step - loss: 11.4977 - categorical_accuracy: 0.1292 - val_loss: 11.3293 - val_categorical_accuracy: 0.1562
Epoch 6/10
30/30 [==============================] - 0s 2ms/step - loss: 11.4686 - categorical_accuracy: 0.1281 - val_loss: 11.2863 - val_categorical_accuracy: 0.1146
Epoch 7/10
30/30 [==============================] - 0s 2ms/step - loss: 11.4305 - categorical_accuracy: 0.1385 - val_loss: 11.1960 - val_categorical_accuracy: 0.1354
Epoch 8/10
30/30 [==============================] - 0s 2ms/step - loss: 11.4499 - categorical_accuracy: 0.1427 - val_loss: 11.2831 - val_categorical_accuracy: 0.0938
Epoch 9/10
30/30 [==============================] - 0s 2ms/step - loss: 11.4924 - categorical_accuracy: 0.1542 - val_loss: 11.4720 - val_categorical_accuracy: 0.0729
Epoch 10/10
30/30 [==============================] - 0s 2ms/step - loss: 11.4732 - categorical_accuracy: 0.1458 - val_loss: 11.1439 - val_categorical_accuracy: 0.1354
<tensorflow.python.keras.callbacks.History object at 0x0000023FA29CA940>
>>> test_x = np.random.random((1000, 72))
>>> test_y = np.random.random((1000, 10))
>>> model.evaluate(test_x, test_y, batch_size=32)
1000/1000 [==============================] - 0s 72us/step
[11.457662460327148, 0.089]
>>> test_data = tf.data.Dataset.from_tensor_slices((test_x, test_y))
>>> test_data = test_data.batch(32).repeat()
>>> model.evaluate(test_data, steps=30)
30/30 [==============================] - 0s 2ms/step
[11.464777660369872, 0.08854166666666667]
>>> result = model.predict(test_x, batch_size=32)
>>> print(result)
[[0.09794696 0.09373893 0.11868412 ... 0.09225973 0.09039019 0.10384268]
 [0.09404054 0.10194635 0.09458948 ... 0.10522062 0.10055067 0.10062526]
 [0.09881989 0.10528595 0.10590459 ... 0.09563414 0.09249627 0.10422694]
 ...
 [0.10042354 0.11020223 0.10501476 ... 0.08610034 0.09960221 0.10058936]
 [0.10001418 0.10045603 0.10777664 ... 0.10316661 0.10119378 0.09854537]
 [0.09882984 0.09552816 0.09732161 ... 0.09852998 0.10665993 0.10750171]]
>>>
```

# 2.2学习笔记

#### tf2.0运行结果

```python

(base) C:\Users\liuxuechao>activate tensorflow2.0

(tensorflow2.0) C:\Users\liuxuechao>python
Python 3.7.6 (default, Jan  8 2020, 20:23:39) [MSC v.1916 64 bit (AMD64)] :: Anaconda, Inc. on win32
Type "help", "copyright", "credits" or "license" for more information.
>>> import tensorflow as a
2020-02-01 21:53:04.841726: W tensorflow/stream_executor/platform/default/dso_loader.cc:55] Could not load dynamic library 'cudart64_101.dll'; dlerror: cudart64_101.dll not found
2020-02-01 21:53:04.845849: I tensorflow/stream_executor/cuda/cudart_stub.cc:29] Ignore above cudart dlerror if you do not have a GPU set up on your machine.
>>> a=3
>>> w=a.Variable([[0.5,1.0]])
Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
AttributeError: 'int' object has no attribute 'Variable'
>>> exit()

(tensorflow2.0) C:\Users\liuxuechao>python
Python 3.7.6 (default, Jan  8 2020, 20:23:39) [MSC v.1916 64 bit (AMD64)] :: Anaconda, Inc. on win32
Type "help", "copyright", "credits" or "license" for more information.
>>> import tensorflow as tf
2020-02-01 21:56:33.716915: W tensorflow/stream_executor/platform/default/dso_loader.cc:55] Could not load dynamic library 'cudart64_101.dll'; dlerror: cudart64_101.dll not found
2020-02-01 21:56:33.720778: I tensorflow/stream_executor/cuda/cudart_stub.cc:29] Ignore above cudart dlerror if you do not have a GPU set up on your machine.
>>> a=3
>>> w=tf.Variable([[0.5,1.0]])
2020-02-01 21:56:50.770829: W tensorflow/stream_executor/platform/default/dso_loader.cc:55] Could not load dynamic library 'nvcuda.dll'; dlerror: nvcuda.dll not found
2020-02-01 21:56:50.776085: E tensorflow/stream_executor/cuda/cuda_driver.cc:351] failed call to cuInit: UNKNOWN ERROR (303)
2020-02-01 21:56:50.784561: I tensorflow/stream_executor/cuda/cuda_diagnostics.cc:169] retrieving CUDA diagnostic information for host: DESKTOP-30VSF9G
2020-02-01 21:56:50.789084: I tensorflow/stream_executor/cuda/cuda_diagnostics.cc:176] hostname: DESKTOP-30VSF9G
2020-02-01 21:56:50.794362: I tensorflow/core/platform/cpu_feature_guard.cc:142] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2
>>> x=tf.Variable([[2.0],[1.0]])
>>> y=tf.matmul(w,x)
>>> init_op=tf.global_variables_initializer()
Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
AttributeError: module 'tensorflow' has no attribute 'global_variables_initializer'
>>> exit()

(tensorflow2.0) C:\Users\liuxuechao>python -c "import tensorflow as tf;print(tf.reduce_sum(tf.random.normal([1000, 1000])))"
2020-02-01 22:00:28.463462: W tensorflow/stream_executor/platform/default/dso_loader.cc:55] Could not load dynamic library 'cudart64_101.dll'; dlerror: cudart64_101.dll not found
2020-02-01 22:00:28.468114: I tensorflow/stream_executor/cuda/cudart_stub.cc:29] Ignore above cudart dlerror if you do not have a GPU set up on your machine.
2020-02-01 22:00:30.902134: W tensorflow/stream_executor/platform/default/dso_loader.cc:55] Could not load dynamic library 'nvcuda.dll'; dlerror: nvcuda.dll not found
2020-02-01 22:00:30.907287: E tensorflow/stream_executor/cuda/cuda_driver.cc:351] failed call to cuInit: UNKNOWN ERROR (303)
2020-02-01 22:00:30.917180: I tensorflow/stream_executor/cuda/cuda_diagnostics.cc:169] retrieving CUDA diagnostic information for host: DESKTOP-30VSF9G
2020-02-01 22:00:30.921140: I tensorflow/stream_executor/cuda/cuda_diagnostics.cc:176] hostname: DESKTOP-30VSF9G
2020-02-01 22:00:30.923685: I tensorflow/core/platform/cpu_feature_guard.cc:142] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2
tf.Tensor(1303.781, shape=(), dtype=float32)

(tensorflow2.0) C:\Users\liuxuechao>python
Python 3.7.6 (default, Jan  8 2020, 20:23:39) [MSC v.1916 64 bit (AMD64)] :: Anaconda, Inc. on win32
Type "help", "copyright", "credits" or "license" for more information.
>>> import tensorflow as tf
2020-02-01 22:01:49.913307: W tensorflow/stream_executor/platform/default/dso_loader.cc:55] Could not load dynamic library 'cudart64_101.dll'; dlerror: cudart64_101.dll not found
2020-02-01 22:01:49.918716: I tensorflow/stream_executor/cuda/cudart_stub.cc:29] Ignore above cudart dlerror if you do not have a GPU set up on your machine.
>>> from tensorflow.keras import layers
>>> print(tf.__version__)
2.1.0
>>> print(tf.keras.__version__)
2.2.4-tf
>>> model = tf.keras.Sequential()
>>> model.add(layers.Dense(32, activation='relu'))
>>> model.add(layers.Dense(32, activation='relu'))
>>> model.add(layers.Dense(10, activation='softmax'))
>>> layers.Dense(32, activation='sigmoid')
<tensorflow.python.keras.layers.core.Dense object at 0x00000276EDD6F0C8>
>>>  layers.Dense(32, activation=tf.sigmoid)
  File "<stdin>", line 1
    layers.Dense(32, activation=tf.sigmoid)
    ^
IndentationError: unexpected indent
>>> layers.Dense(32, activation=tf.sigmoid)
<tensorflow.python.keras.layers.core.Dense object at 0x00000276EDD43EC8>
>>> layers.Dense(32, kernel_initializer='orthogonal')
<tensorflow.python.keras.layers.core.Dense object at 0x00000276EDD6FEC8>
>>> layers.Dense(32, kernel_initializer=tf.keras.initializers.glorot_normal)
<tensorflow.python.keras.layers.core.Dense object at 0x00000276EDD43EC8>
>>> layers.Dense(32, kernel_regularizer=tf.keras.regularizers.l2(0.01))
<tensorflow.python.keras.layers.core.Dense object at 0x00000276EDD6FEC8>
>>> layers.Dense(32, kernel_regularizer=tf.keras.regularizers.l1(0.01))
<tensorflow.python.keras.layers.core.Dense object at 0x00000276EDD6FFC8>
>>> model = tf.keras.Sequential()
>>> model.add(layers.Dense(32, activation='relu'))
>>> model.add(layers.Dense(32, activation='relu'))
>>> model.add(layers.Dense(10, activation='softmax'))
>>> model.compile(optimizer=tf.keras.optimizers.Adam(0.001),
...
KeyboardInterrupt
>>>  model.compile(optimizer=tf.keras.optimizers.Adam(0.001),
  File "<stdin>", line 1
    model.compile(optimizer=tf.keras.optimizers.Adam(0.001),
    ^
IndentationError: unexpected indent
>>> model.compile(optimizer=tf.keras.optimizers.Adam(0.001),
... loss=tf.keras.losses.categorical_crossentropy,
...               metrics=[tf.keras.metrics.categorical_accuracy])
>>> import numpy as np
>>> train_x = np.random.random((1000, 72))
>>> train_y = np.random.random((1000, 10))
>>> val_x = np.random.random((200, 72))
>>> val_y = np.random.random((200, 10))
>>> model.fit(train_x, train_y, epochs=10, batch_size=100,
...            validation_data=(val_x, val_y))
2020-02-01 22:08:54.750753: W tensorflow/stream_executor/platform/default/dso_loader.cc:55] Could not load dynamic library 'nvcuda.dll'; dlerror: nvcuda.dll not found
2020-02-01 22:08:54.756184: E tensorflow/stream_executor/cuda/cuda_driver.cc:351] failed call to cuInit: UNKNOWN ERROR (303)
2020-02-01 22:08:54.765903: I tensorflow/stream_executor/cuda/cuda_diagnostics.cc:169] retrieving CUDA diagnostic information for host: DESKTOP-30VSF9G
2020-02-01 22:08:54.770802: I tensorflow/stream_executor/cuda/cuda_diagnostics.cc:176] hostname: DESKTOP-30VSF9G
2020-02-01 22:08:54.778288: I tensorflow/core/platform/cpu_feature_guard.cc:142] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2
Train on 1000 samples, validate on 200 samples
Epoch 1/10
1000/1000 [==============================] - 0s 425us/sample - loss: 11.8297 - categorical_accuracy: 0.0920 - val_loss: 11.9879 - val_categorical_accuracy: 0.1050
Epoch 2/10
1000/1000 [==============================] - 0s 20us/sample - loss: 11.9880 - categorical_accuracy: 0.0940 - val_loss: 12.2945 - val_categorical_accuracy: 0.1000
Epoch 3/10
1000/1000 [==============================] - 0s 21us/sample - loss: 12.4026 - categorical_accuracy: 0.0870 - val_loss: 12.8994 - val_categorical_accuracy: 0.1100
Epoch 4/10
1000/1000 [==============================] - 0s 23us/sample - loss: 13.1244 - categorical_accuracy: 0.0880 - val_loss: 13.8565 - val_categorical_accuracy: 0.1200
Epoch 5/10
1000/1000 [==============================] - 0s 21us/sample - loss: 14.3370 - categorical_accuracy: 0.0820 - val_loss: 15.4769 - val_categorical_accuracy: 0.1150
Epoch 6/10
1000/1000 [==============================] - 0s 21us/sample - loss: 16.3155 - categorical_accuracy: 0.0840 - val_loss: 17.9826 - val_categorical_accuracy: 0.1200
Epoch 7/10
1000/1000 [==============================] - 0s 21us/sample - loss: 19.1181 - categorical_accuracy: 0.0920 - val_loss: 21.3029 - val_categorical_accuracy: 0.1200
Epoch 8/10
1000/1000 [==============================] - 0s 19us/sample - loss: 22.7502 - categorical_accuracy: 0.0900 - val_loss: 25.6009 - val_categorical_accuracy: 0.1000
Epoch 9/10
1000/1000 [==============================] - 0s 23us/sample - loss: 27.5551 - categorical_accuracy: 0.0950 - val_loss: 31.1916 - val_categorical_accuracy: 0.1100
Epoch 10/10
1000/1000 [==============================] - 0s 20us/sample - loss: 33.3495 - categorical_accuracy: 0.0880 - val_loss: 37.4008 - val_categorical_accuracy: 0.1150
<tensorflow.python.keras.callbacks.History object at 0x00000276EEE60348>
>>> dataset = tf.data.Dataset.from_tensor_slices((train_x, train_y))
>>> dataset = dataset.batch(32)
>>> dataset = dataset.repeat()
>>> val_dataset = tf.data.Dataset.from_tensor_slices((val_x, val_y))
>>> val_dataset = val_dataset.batch(32)
>>> val_dataset = val_dataset.repeat()
>>> model.fit(dataset, epochs=10, steps_per_epoch=30,
...            validation_data=val_dataset,
...            validation_data=val_dataset, validation_steps=3)
  File "<stdin>", line 3
SyntaxError: keyword argument repeated
>>> model.fit(dataset, epochs=10, steps_per_epoch=30,
...            validation_data=val_dataset, validation_steps=3)
Train for 30 steps, validate for 3 steps
Epoch 1/10
30/30 [==============================] - 0s 8ms/step - loss: 47.8062 - categorical_accuracy: 0.0833 - val_loss: 62.9789 - val_categorical_accuracy: 0.1146
Epoch 2/10
30/30 [==============================] - 0s 2ms/step - loss: 75.9359 - categorical_accuracy: 0.0929 - val_loss: 96.5823 - val_categorical_accuracy: 0.1042
Epoch 3/10
30/30 [==============================] - 0s 2ms/step - loss: 110.9199 - categorical_accuracy: 0.0919 - val_loss: 132.9336 - val_categorical_accuracy: 0.1146
Epoch 4/10
30/30 [==============================] - 0s 2ms/step - loss: 145.5147 - categorical_accuracy: 0.0962 - val_loss: 169.6907 - val_categorical_accuracy: 0.1042
Epoch 5/10
30/30 [==============================] - 0s 1ms/step - loss: 180.9468 - categorical_accuracy: 0.1026 - val_loss: 204.7337 - val_categorical_accuracy: 0.1146
Epoch 6/10
30/30 [==============================] - 0s 2ms/step - loss: 213.9926 - categorical_accuracy: 0.0876 - val_loss: 238.1935 - val_categorical_accuracy: 0.0938
Epoch 7/10
30/30 [==============================] - 0s 2ms/step - loss: 241.0229 - categorical_accuracy: 0.0962 - val_loss: 265.0216 - val_categorical_accuracy: 0.1146
Epoch 8/10
30/30 [==============================] - 0s 2ms/step - loss: 259.1919 - categorical_accuracy: 0.0940 - val_loss: 275.2800 - val_categorical_accuracy: 0.1042
Epoch 9/10
30/30 [==============================] - 0s 2ms/step - loss: 255.9588 - categorical_accuracy: 0.1058 - val_loss: 260.6051 - val_categorical_accuracy: 0.1146
Epoch 10/10
30/30 [==============================] - 0s 2ms/step - loss: 231.3891 - categorical_accuracy: 0.1004 - val_loss: 233.5428 - val_categorical_accuracy: 0.0938
<tensorflow.python.keras.callbacks.History object at 0x00000276EEE60A88>
>>> test_x = np.random.random((1000, 72))
>>> test_y = np.random.random((1000, 10))
>>> model.evaluate(test_x, test_y, batch_size=32)
1000/1000 [==============================] - 0s 87us/sample - loss: 217.4756 - categorical_accuracy: 0.1070
[217.47558471679687, 0.107]
>>> test_data = tf.data.Dataset.from_tensor_slices((test_x, test_y))
>>> test_data = test_data.batch(32).repeat()
>>> model.evaluate(test_data, steps=30)
30/30 [==============================] - 0s 830us/step - loss: 217.5327 - categorical_accuracy: 0.1083
[217.53268330891927, 0.108333334]
>>> result = model.predict(test_x, batch_size=32)
>>> print(result)
[[0.         0.09408058 0.18456095 ... 0.00312631 0.0528776  0.5596742 ]
 [0.         0.07900902 0.16783477 ... 0.00502137 0.08891902 0.5420419 ]
 [0.         0.12148204 0.29399288 ... 0.00643831 0.1307136  0.32100162]
 ...
 [0.         0.0841686  0.24969104 ... 0.00718069 0.08367687 0.44839454]
 [0.         0.06567922 0.3431278  ... 0.00341688 0.10024671 0.41551104]
 [0.         0.05106765 0.16744436 ... 0.00483427 0.0938361  0.5832979 ]]
>>>
```

可见运行结果略有差异

# 

# csv数据处理

数据预处理：

所有的特征都是用一个热编码数字的。对这些特征进行缩放以避免具有可能在结果中权重过大的值的特征。
用决策树对数据集KDD做分析
https://github.com/CynthiaKoopman/Network-Intrusion-Detection/blob/master/DecisionTree_IDS.ipynb

```python
df_test = pd.read_csv("KDDTest+_2.csv", header=None, names = col_names)
#文件如果没有放在同一个文件夹之下，需要加全路径
#header为指定的行作为表头
#指定表头
```



```python
df.describe()
#未出结果
```

```python
print(df['label'].value_counts())
#value_counts()函数 
#value_counts函数用于统计dataframe或series中不同数或字符串出现的次数
#ascending=True时,按升序排列.
#normalize=True时,可计算出不同字符出现的频率,画柱状图统计时可以用到.
```

​	one-hot

因此，首先需要使用LabelEncoder转换特性，将每个类别转换为一个数字。

“这个转换器的输入应该是一个整数矩阵，表示类别（离散）特征所接受的值。输出将是一个稀疏矩阵，其中每列对应一个特征的一个可能值。假设输入特征的值在[0，n_值]范围内

```python
print('Training set:')
for col_name in df.columns:
    if df[col_name].dtypes == 'object' :
#.dtypes:数据类型
        unique_cat = len(df[col_name].unique()) #.unique去掉重复值
        print("Feature '{col_name}' has {unique_cat} categories".format(col_name=col_name, unique_cat=unique_cat))

#see how distributed the feature service is, it is evenly distributed and therefore we need to make dummies for all.
print()
print('Distribution of categories in service:')
print(df['service'].value_counts().sort_values(ascending=False).head())
```

# 可调参数

1.图片形状

# 第三周工作

0，已完成：CNN实现二分类                      #（个人问题:数据处理过程代码不是很清楚）**必须清楚！**

1.接下来工作目标：

​    0.CNN调参（重点调数据量比较少的的结构，以用于生成器配合）

​    ######1.多分类的实现，（标签改，输出）

​     2.利用生成器产生样本

# DCGAN

```python
# Batch and shuffle the data
train_dataset = tf.data.Dataset.from_tensor_slices #他的所用是切分传入的 Tensor 的第一个维度，生成相应的 dataset 。
(train_images).shuffle(BUFFER_SIZE).batch(BATCH_SIZE)
```

![1584195556821](C:\Users\liuxuechao\AppData\Roaming\Typora\typora-user-images\1584195556821.png)

有一个`shuffle`方法，用来打乱数据集中数据顺序，训练时非常常用。

shuffle是防止数据过拟合的重要手段，

Batch Size定义：一次训练所选取的样本数。

1、没有Batch Size，梯度准确，只适用于小样本数据库 
2、Batch Size=1，梯度变来变去，非常不准确，网络很难收敛。 
3、Batch Size增大，梯度变准确， 
4、Batch Size增大，梯度已经非常准确，再增加Batch Size也没有用

注意：Batch Size增大了，要到达相同的准确度，必须要增大epoch。

```python
train_images = train_images.reshape(train_images.shape[0], 28, 28, 1).astype('float32')
train_images = (train_images - 127.5) / 127.5 # Normalize the images to [-1, 1]
```

![1584372201797](C:\Users\liuxuechao\AppData\Roaming\Typora\typora-user-images\1584372201797.png)

![1584372362280](C:\Users\liuxuechao\AppData\Roaming\Typora\typora-user-images\1584372362280.png)

```python
(layers.Dense
```

inputs：输入该网络层的数据

units：输出的维度大小，改变inputs的最后一维

activation：激活函数，即神经网络的非线性变化

use_bias：使用bias为True（默认使用），不用bias改成False即可，是否使用偏置项

trainable=True:表明该层的参数是否参与训练。如果为真则变量加入到图集合中
![1584375108006](C:\Users\liuxuechao\AppData\Roaming\Typora\typora-user-images\1584375108006.png)

https://blog.csdn.net/theonegis/article/details/80115340

![1584375388329](C:\Users\liuxuechao\AppData\Roaming\Typora\typora-user-images\1584375388329.png)

batchnormalization

 https://zhuanlan.zhihu.com/p/113233908 

# 多分类之数据处理

![1584428379562](C:\Users\liuxuechao\AppData\Roaming\Typora\typora-user-images\1584428379562.png)

![1584434156163](C:\Users\liuxuechao\AppData\Roaming\Typora\typora-user-images\1584434156163.png)

#### 问题

1.添加缺失的几行时打乱了顺序，应该加到对应的位置

2.conda安装sklearn 用conda install scikit-learn

3.numpy和tensor之间转化

https://blog.csdn.net/weixin_44473755/article/details/102939703