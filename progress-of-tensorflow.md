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

# 2.24框架函数研究

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



