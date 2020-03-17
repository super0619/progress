

## problem. 

## jupyter notebook in tensorflow不能使用，出现kernel error 

 jupyter notebook in tensorflow不能使用，出现kernel error 

```python
Traceback (most recent call last):
  File "C:\Users\liuxuechao\Anaconda3\envs\tensorflow\lib\site-packages\tornado\web.py", line 1592, in _execute
    result = yield result
  File "C:\Users\liuxuechao\Anaconda3\envs\tensorflow\lib\site-packages\tornado\gen.py", line 1133, in run
    value = future.result()
  File "C:\Users\liuxuechao\Anaconda3\envs\tensorflow\lib\asyncio\futures.py", line 294, in result
    raise self._exception
  File "C:\Users\liuxuechao\Anaconda3\envs\tensorflow\lib\site-packages\tornado\gen.py", line 1141, in run
    yielded = self.gen.throw(*exc_info)
  File "C:\Users\liuxuechao\Anaconda3\envs\tensorflow\lib\site-packages\notebook\services\sessions\handlers.py", line 73, in post
    type=mtype))
  File "C:\Users\liuxuechao\Anaconda3\envs\tensorflow\lib\site-packages\tornado\gen.py", line 1133, in run
    value = future.result()
  File "C:\Users\liuxuechao\Anaconda3\envs\tensorflow\lib\asyncio\futures.py", line 294, in result
    raise self._exception
  File "C:\Users\liuxuechao\Anaconda3\envs\tensorflow\lib\site-packages\tornado\gen.py", line 1141, in run
    yielded = self.gen.throw(*exc_info)
  File "C:\Users\liuxuechao\Anaconda3\envs\tensorflow\lib\site-packages\notebook\services\sessions\sessionmanager.py", line 79, in create_session
    kernel_id = yield self.start_kernel_for_session(session_id, path, name, type, kernel_name)
  File "C:\Users\liuxuechao\Anaconda3\envs\tensorflow\lib\site-packages\tornado\gen.py", line 1133, in run
    value = future.result()
  File "C:\Users\liuxuechao\Anaconda3\envs\tensorflow\lib\asyncio\futures.py", line 294, in result
    raise self._exception
  File "C:\Users\liuxuechao\Anaconda3\envs\tensorflow\lib\site-packages\tornado\gen.py", line 1141, in run
    yielded = self.gen.throw(*exc_info)
  File "C:\Users\liuxuechao\Anaconda3\envs\tensorflow\lib\site-packages\notebook\services\sessions\sessionmanager.py", line 92, in start_kernel_for_session
    self.kernel_manager.start_kernel(path=kernel_path, kernel_name=kernel_name)
  File "C:\Users\liuxuechao\Anaconda3\envs\tensorflow\lib\site-packages\tornado\gen.py", line 1133, in run
    value = future.result()
  File "C:\Users\liuxuechao\Anaconda3\envs\tensorflow\lib\asyncio\futures.py", line 294, in result
    raise self._exception
  File "C:\Users\liuxuechao\Anaconda3\envs\tensorflow\lib\site-packages\tornado\gen.py", line 326, in wrapper
    yielded = next(result)
  File "C:\Users\liuxuechao\Anaconda3\envs\tensorflow\lib\site-packages\notebook\services\kernels\kernelmanager.py", line 160, in start_kernel
    super(MappingKernelManager, self).start_kernel(**kwargs)
  File "C:\Users\liuxuechao\Anaconda3\envs\tensorflow\lib\site-packages\jupyter_client\multikernelmanager.py", line 110, in start_kernel
    km.start_kernel(**kwargs)
  File "C:\Users\liuxuechao\Anaconda3\envs\tensorflow\lib\site-packages\jupyter_client\manager.py", line 240, in start_kernel
    self.write_connection_file()
  File "C:\Users\liuxuechao\Anaconda3\envs\tensorflow\lib\site-packages\jupyter_client\connect.py", line 547, in write_connection_file
    kernel_name=self.kernel_name
  File "C:\Users\liuxuechao\Anaconda3\envs\tensorflow\lib\site-packages\jupyter_client\connect.py", line 212, in write_connection_file
    with secure_write(fname) as f:
  File "C:\Users\liuxuechao\Anaconda3\envs\tensorflow\lib\contextlib.py", line 59, in __enter__
    return next(self.gen)
  File "C:\Users\liuxuechao\Anaconda3\envs\tensorflow\lib\site-packages\jupyter_client\connect.py", line 100, in secure_write
    win32_restrict_file_to_user(fname)
  File "C:\Users\liuxuechao\Anaconda3\envs\tensorflow\lib\site-packages\jupyter_client\connect.py", line 53, in win32_restrict_file_to_user
    import win32api
ImportError: No module named 'win32api'

```

处理：1.重装jupyter

​           2.下载win32api库均未成功

## problem.

##  "Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 FMA"

1.忽略

```python
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
```

 

2.编译tensorflow源代码

# 2.2更新

## problem

## tensorflow2.1.0安装问题

<img src="C:\Users\liuxuechao\AppData\Roaming\Typora\typora-user-images\1580567463151.png" alt="1580567463151" style="zoom:50%;" />

持续报错：dll load failed :找不到指定的模块

#### 检查过程：

1.检查了pillow，无效

2.检查了anaconda 环境变量

3.尝试重装tensorflow

4.重新阅读安装指导文档，发现msvcp140_1.dll文档缺失，问题解决

 deb ![img](file:///C:\Users\liuxuechao\AppData\Local\Temp\%W@GJ$ACOF(TYDYECOKVDYB.png)http://mirrors.aliyun.com/ubuntu/ bionic main restricted universe multiverse
deb ![img](file:///C:\Users\liuxuechao\AppData\Local\Temp\%W@GJ$ACOF(TYDYECOKVDYB.png)http://mirrors.aliyun.com/ubuntu/ bionic-security main restricted universe multiverse
deb ![img](file:///C:\Users\liuxuechao\AppData\Local\Temp\%W@GJ$ACOF(TYDYECOKVDYB.png)http://mirrors.aliyun.com/ubuntu/ bionic-updates main restricted universe multiverse
deb ![img](file:///C:\Users\liuxuechao\AppData\Local\Temp\%W@GJ$ACOF(TYDYECOKVDYB.png)http://mirrors.aliyun.com/ubuntu/ bionic-proposed main restricted universe multiverse
deb ![img](file:///C:\Users\liuxuechao\AppData\Local\Temp\%W@GJ$ACOF(TYDYECOKVDYB.png)http://mirrors.aliyun.com/ubuntu/ bionic-backports main restricted universe multiverse
deb-src ![img](file:///C:\Users\liuxuechao\AppData\Local\Temp\%W@GJ$ACOF(TYDYECOKVDYB.png)http://mirrors.aliyun.com/ubuntu/ bionic main restricted universe multiverse
deb-src ![img](file:///C:\Users\liuxuechao\AppData\Local\Temp\%W@GJ$ACOF(TYDYECOKVDYB.png)http://mirrors.aliyun.com/ubuntu/ bionic-security main restricted universe multiverse
deb-src ![img](file:///C:\Users\liuxuechao\AppData\Local\Temp\%W@GJ$ACOF(TYDYECOKVDYB.png)http://mirrors.aliyun.com/ubuntu/ bionic-updates main restricted universe multiverse
deb-src ![img](file:///C:\Users\liuxuechao\AppData\Local\Temp\%W@GJ$ACOF(TYDYECOKVDYB.png)http://mirrors.aliyun.com/ubuntu/ bionic-proposed main restricted universe multiverse
deb-src ![img](file:///C:\Users\liuxuechao\AppData\Local\Temp\%W@GJ$ACOF(TYDYECOKVDYB.png)http://mirrors.aliyun.com/ubuntu/ bionic-backports main restricted universe multiverse
