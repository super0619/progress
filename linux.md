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

