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

# 2.1学习笔记

### 层堆叠模型试验 tf.keras.Sequential 模型 

keras框架更加注重深度学习模型构建，提供了便利的API接口，简单堆积即可实现

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

