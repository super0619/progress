# 1.换源问题

将/etc/apt/sources.list中换为阿里云源

百度Ubuntu18.04阿里云源

cd /etc/apt

vi sources.list

i编辑

2dd 删除

：wq!强制保存并退出

apt-get update

apt-get upgrade



# 2.vmtools

进入底层安装包

./vmware-install.pl



# 常见命令

## 补充：/根目录  ./当前目录

## --help

## ls 

### ls -a

显示文件及文件夹，包括隐藏文件

### ls -l

目录文件夹的详细信息

### ls -al

### ls test/

查看当前目录下test文件夹中的内容

## cd

### cd ../

返回上一级

### cd /

linux根目录

### cd /home/super/

tab可以补全

## pwd

显示当前路径

## uname

### uname -a

全部信息

## clear

清屏

## cat

cat 文件路径

查看文件内容

## touch

touch a.c

在当前目录下创建文件a.c

touch test/a.c

在当前目录下的test文件夹中创建a.c

## cp

cp a.c b.c

在当前目录下copy a.c 成为b.c

## rm

删除文件

### rm test/ -rf

文件夹test（test就是一个目录）删除

## mkdir

创建文件夹

## rmdir

rmdir test/

删除目录

## mv

### mv a.c b.c

将a.c 重命名为b.c

### mv test/ test1/

目录test改名为test1

mv a.c test1/

将a.c移动到当前目录下test1文件夹中

## ifconfig

显示网络配置

![1581561340077](C:\Users\liuxuechao\AppData\Roaming\Typora\typora-user-images\1581561340077.png)

## reboot

重启

## man

系统帮助命令，查看命令

q退出

## sync

数据同步写入磁盘

## find

### find -name a.c

查找当前目录下所有文件

## grap

查找字符串

### -nir

## du

察看文件大小

du test/ -sh

-h以人类可读方式

-s不显示旗下子目录

## gedit

编辑文档，文件

## ps

显示行程

###  ps -aux

全部

## top

同win任务管理器

## file

查看文件类型

# 软件安装

## apt安装

解决依赖关系

## deb软件包安装

.deb=.exe

进入相应目录

双击或sudo dpkg -i xxx.deb

应用图标一般在/usr/share/applications里面，复制到桌面

## 编译安装

make

make install

![1581568340570](C:\Users\liuxuechao\AppData\Roaming\Typora\typora-user-images\1581568340570.png)

# 文件系统结构

<img src="C:\Users\liuxuechao\AppData\Roaming\Typora\typora-user-images\1581569940730.png" alt="1581569940730" style="zoom:50%;" />

![1581570973494](C:\Users\liuxuechao\AppData\Roaming\Typora\typora-user-images\1581570973494.png)