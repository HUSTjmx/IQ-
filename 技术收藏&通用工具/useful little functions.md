# useful little functions

​	在编写着色器时或在任何过程创建过程中（纹理，建模，着色，动画...），您通常会发现自己以不同的方式修改信号，以使其表现出您想要的方式。通常使用==smoothstep==阈值化一些值，或使用==pow==整形信号，使用==clamp==进行裁剪，使用==fmod==进行重复，使用==mix==进行混合，使用==exp==进行衰减，等等 。这些功能很方便，因为在大多数系统中默认情况下，您可以使用它们作为硬件指令或语言中的函数调用。但是，有些经常使用的操作以您仍然经常使用的任何语言都不存在。您是否发现自己要减去==smoothstep==来隔离某个范围或创建环？还是执行一些平滑的裁切操作以避免被大数除法？



### Almost Identity(I)

![](https://jmx-paper.oss-cn-beijing.aliyuncs.com/IQ%E5%A4%A7%E7%A5%9E%E5%8D%9A%E5%AE%A2%E9%98%85%E8%AF%BB/%E5%9B%BE%E7%89%87/useful%20little%20functions/AlmostIdentity%28I%29.png)

​	想象一下，除非它为零或非常接近它，否则您不希望更改它的值，在这种情况下，您想用一个小的常数替换该值。那么，与其做一个引入不连续的条件分支，不如将你的值与你的阈值平滑地融合在一起。让m是阈值（m以上的东西保持不变），n是当你的输入为零时，将采取的值。那么，下面的函数就可以进行软剪裁（以立方体的方式）。

```c#
float almostIdentity( float x, float m, float n )
{
    if( x>m ) return x;
    const float a = 2.0*n - m;
    const float b = 2.0*m - 3.0*n;
    const float t = x/m;
    return (a*t + b)*t*t + n;
}
```





### Almost Unit Identity

![](https://jmx-paper.oss-cn-beijing.aliyuncs.com/IQ%E5%A4%A7%E7%A5%9E%E5%8D%9A%E5%AE%A2%E9%98%85%E8%AF%BB/%E5%9B%BE%E7%89%87/useful%20little%20functions/AlmostUnitIdentity.png)

