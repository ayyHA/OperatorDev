# OperatorDev
**算子开发**

## x86

### OptimizeGEMM

这个文件夹下面的优化GEMM方法是跟着flame/how-to-optimize-gemm这个项目进行学习的第一个算子优化的项目(并没有优化到极致)
本项目是针对列主序的通用矩阵乘进行优化,主要包括:

- 朴素的矩阵乘的一次内积变做通过调用内积函数add_dot,一次做四次内积add_dot1x4,修改步距以达成;
- 将内积函数add_dot内联到add_dot1x4中,减少函数调用,并放置在一个for中,减少循环次数,同时利用空间局部性;
- 采用寄存器来存储频繁使用的数据,对B阵则将宏替换换作指针寻址,减少浮点操作次数;
- 对B阵的指针寻址换做间接寻址,循环展开;

以上是将1次内积变为4次内积,并作出的优化手段,接着在4次内积的基础上优化为4x4,共16次内积的优化:

- 采用寄存器将频繁使用的值进行存储;
- 采用intrinsic编程(SSE指令集);
- 将矩阵分块为A(256\*128),B(128\*1024),并对A阵的分块内容进行打包,存储在连续的内存空间;
- 将B阵的分块也进行打包,同时避免重复的数据获取

## neon

Neon是ARM支持的一款高级SIMD,本仓库的Neon开发是基于aarch64进行的

### RGB deinterleaving

这是arm教程的一个例子,对于RGB三色通道分离采用neon intrinsic操作进行优化

### matrix mutiplication

这是arm教程的一个例子,对于矩阵乘采用分块进行优化,其中对4x4的小矩阵作为小块进行neon intrinsic优化

### array_weighted_sum

这是一个数组加权和的例子,用于利用neon intrinsic进行优化,同时利用neon assembly进行优化

### box_filter

这是一个盒子滤波的优化算法,从算法的朴素实现\-\>行列分离\-\>利用opencv的优化算法进行优化\-\>采用neon intrinsic使之矢量化\-\>采用neon assembly重写关键算法.分步骤实现优化

## RISC-V

此目录下仅有一个`hello_world.cpp`,该文件主要是利用RISC-V进行内联汇编的简单编程,由于缺乏RISC-V的硬件支持,相关实验暂无从展开,且RISC-V是本人进行汇编学习的entry而非目的,因此该部分会较久进行一次内容更新.