# IQ大神博客阅读心得7

| 名称                              | 概述              |
| --------------------------------- | ----------------- |
| [Popcorn Images](#Popcorn-Images) | Pop图片的生成方法 |
| [Domain Warping](#Domain_Warping) |                   |
|                                   |                   |
|                                   |                   |
|                                   |                   |
|                                   |                   |
|                                   |                   |
|                                   |                   |
|                                   |                   |
|                                   |                   |
|                                   |                   |







## Popcorn Images

这些所谓的“爆米花”图片是由Cliff Pickover很久以前创作的。他们背后的想法是绘制一个动态系统的演化图。
$$
p^`(t)=v(p)
$$
其中p是一个点，v是一个静止的速度场。要在计算机上进行这样的模拟，最简单的方法是编写一个简单的欧拉积分器，用它来模拟具有一些差分和小脉冲时间（a small delta time）的导数。在二维空间中，它是这样的
$$
x_{n+1}=x_n+\lambda \cdot f(x,y)\\
y_{n+1}=y_n+\lambda\cdot f(x,y)
$$
其中λ是时间步长值（应该是小的）。现在就看你怎么选择f(p)和g(p)的公式了。最初的Pickover的公式是三角函数，但你可以输入任何你喜欢的东西。在我1999年的实验中，我使用了Pickover的原始公式。结果被录入了64kb的demo，叫做rare。

后来在2006年我做了这个视频，在这个视频中，我把f(p)和g(p)随着时间的推移做了动画。我再次使用了三角函数(类似于Pickover的函数)，因为它们产生了一些漂亮的类似于流体的形状。这段视频分为四个不同的部分，每个部分有一个不同的公式
$$
f(x,y)=cos(t_0+y+cos(t_1+\pi x))\\
g(x,y)=cos(t_2+x+cos(t_3+\pi y))
$$
The ***ti*** parameters are linearly time varying values that produce the actual animation.

与非常类似的IFS方法一样，产生图像的方法是选择一个随机点，用上面的公式进行迭代。这就产生了一个空间上的轨道，即二维的平面）。) 人们必须通过计算一个迭代点落在平面上的多少次，来跟踪平面上每个像素点的密度。在执行了大约10亿次迭代之后，图像应该是足够无噪声的。然后就看你如何通过一些你喜欢的调色板来解释密度了。在[我的例子](https://www.shadertoy.com/view/Wss3zB)中，我的做法是用略微不同的参数计算了三次密度，然后将得到的密度分配给图像的红、绿、蓝通道。

![](https://jmx-paper.oss-cn-beijing.aliyuncs.com/IQ%E5%A4%A7%E7%A5%9E%E5%8D%9A%E5%AE%A2%E9%98%85%E8%AF%BB/%E5%9B%BE%E7%89%87/procedural%20content/Pop.PNG)

[其他](https://www.shadertoy.com/view/Mdl3RH)

以及未展开讲的[Icon Image](https://www.iquilezles.org/www/articles/iconimages/iconimages.htm)





## Domain Warping

