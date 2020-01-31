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