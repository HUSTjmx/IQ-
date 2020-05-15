# IQ大神博客阅读心得7

| 名称                                                         | 概述                         |
| ------------------------------------------------------------ | ---------------------------- |
| [Popcorn Images](#Popcorn-Images)                            | Pop图片的生成方法            |
| [Domain Warping](#Domain_Warping)                            | Warping图的生成方法          |
| [Voronoi Edges](#Voronoi-Edges)                              | 细胞图的生成和边界问题       |
| [Smooth Voronoi](#Smooth-Voronoi)                            | 细胞图不连续性的几个解决方法 |
| [Voronoise](#Voronoise)                                      | 噪声和Voronoi的统一以及实现  |
| [Advanced Value Noise](#Advanced-Value-Noise)                | ValueNoise的分析和使用       |
| [Gradient Noise Derivatives](#Gradient-Noise-Derivatives)    | 梯度噪声导数的实现           |
| [Filtering The Checkerboard Pattern](#Filtering-The-Checkerboard-Pattern) |                              |
|                                                              |                              |
|                                                              |                              |
|                                                              |                              |







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

翘曲或dommain失真是计算机图形学中用于生成过程纹理和几何图形的一种非常常见的技术。它经常被用来捏住一个物体，拉紧它，扭曲它，弯曲它，使它更厚或应用任何变形你想要的。只要你的基本颜色图案或几何图形被定义为空间的函数，它就可以工作。在本文中，我将只展示一个非常特殊的翘曲情况——基于噪音的翘曲或噪音函数。这是从1984年开始使用的，当时Ken Perlin自己创建了他的第一个程序大理石纹理。

==The basics==

假设有一些几何图形或图像被定义为空间的函数。对于几何，它是 f(x,y,z)；对于图像，它是 f(x,y)。我们可以将这两种情况更简洁地写成 f(p)，其中 p 是空间中的位置，for which we can evaluate the volumetric density that will define our (iso)surface or image color。Warping仅仅意味着我们在计算f之前用另一个函数g(p)来扭曲定义域。基本上，我们用 f(g(p)) 来替换 f(p)。g 可以是任何东西，但是我们经常想要对f的图像做一些变形考虑到它的常规行为。然后，让g(p)等于恒等式加上一个任意的小变形h(p)是有意义的，或者换句话说
$$
g(p)=p+h(p)\\
f(p+h(p))
$$
这项技术非常强大，可以让你塑造苹果、建筑物、动物或任何你能想象到的东西。对于本文的目的，我们将只使用基于FBM的模式，包括 f 和 h 。这将生成一些抽象但漂亮的图像，它们具有非常自然的质量。

==The idea==

因此，我们将使用一些标准的fBM（分数布朗运动），这是一组简单的噪声波（增加频率和减少振幅）的总和。在右边的第一个图像中显示了一个简单的fBM。代码看起来是这样的：

```c#
float pattern(in vec2 p)
{
	return fbm(p);
}
```

![](https://jmx-paper.oss-cn-beijing.aliyuncs.com/IQ%E5%A4%A7%E7%A5%9E%E5%8D%9A%E5%AE%A2%E9%98%85%E8%AF%BB/%E5%9B%BE%E7%89%87/procedural%20content/gfx04.jpg)

现在我们可以添加第一个域的扭曲（下图）

![](https://jmx-paper.oss-cn-beijing.aliyuncs.com/IQ%E5%A4%A7%E7%A5%9E%E5%8D%9A%E5%AE%A2%E9%98%85%E8%AF%BB/%E5%9B%BE%E7%89%87/procedural%20content/gfx03.jpg)

```c#
float pattern(in vec2 p)
{
	vec2 q=vec2(fbm(p+vec2(0.0,0.0)),
				fbm(p+vec2(5.2,1.3)));
	return fbm(p+4.0*q);
}
```

注意，我们如何使用两个一维FBM调用来模拟一个二维FBM，这是我们在二维中置换一个点所需要的。最后，我们添加第二个warping（下图）

![](https://jmx-paper.oss-cn-beijing.aliyuncs.com/IQ%E5%A4%A7%E7%A5%9E%E5%8D%9A%E5%AE%A2%E9%98%85%E8%AF%BB/%E5%9B%BE%E7%89%87/procedural%20content/gfx02.jpg)

```c#
float pattern(in vec2 p)
{
	vec2 q=vec2(fbm(p+vec2(0.0,0.0)),
				fbm(p+vec2(5.2,1.3)));
	
	vec2 r=vec2(fbm(p+4.0*q+vec2(1.7,9.2)),
				fbm(p+4.0*q+vec2(8.3,2.8)));
				
	return fbm(p+4.0*r);
}
```

==The experiments==

现在基础设置好了，是时候开始了。第一个明显的想法是引入时间作为参数来获得某种动画。

下一步是添加一些颜色到我们的图像。我们可以简单地将调色板映射到我们的密度值。这是一个良好的开端，但还不够。我们可能想要使用funcion的内部值来获得一些额外的颜色模式和形状。毕竟，我们有三个FBM函数可以改变最终图像的内部结构，所以为什么不使用它们来获得一些额外的颜色呢?我们要做的第一件事，就是将它们传到函数外

```c#
float pattern( in vec2 p, out vec2 q, out vec2 r )
{
    q.x = fbm( p + vec2(0.0,0.0) );
    q.y = fbm( p + vec2(5.2,1.3) );

    r.x = fbm( p + 4.0*q + vec2(1.7,9.2) );
    r.y = fbm( p + 4.0*q + vec2(8.3,2.8) );

    return fbm( p + 4.0*r );
}
```

[测试例子](https://www.shadertoy.com/view/4s23zz)

![](https://jmx-paper.oss-cn-beijing.aliyuncs.com/IQ%E5%A4%A7%E7%A5%9E%E5%8D%9A%E5%AE%A2%E9%98%85%E8%AF%BB/%E5%9B%BE%E7%89%87/procedural%20content/Warping1.gif)







## Voronoi Edges

​	任何使用voronoi序列创造爬行动物皮肤纹理或干地砖的人都知道，当使用F2 - F1并对其进行阈值化时，分隔细胞的线的宽度是不一致的。这是一个恼人的问题。有技术可以缓解这个问题，但它们只是近似的。在这里，我们将找到voronoi噪声的实现，它可以生成数学上完美的细胞分离线，并且是完全程序化的。

==The Problem==

​	从距离第二个最近点的距离中减去到最近点的距离，或者人们所说的*F2-F1 Voronoi*，它非常接近单元格边界生成器。实际上，单元格的边界发生在这两个距离相等的位置（两个最近邻居的等距点），因此函数F2-F1恰好在单元格的边界取值0.0，这非常有用。因此，一个人很容易在两个小数字之间简单地平滑F2-F1，并将其称为“单元边缘”。这种工作，但不是完全。F2-F1并不是真正的距离，因为它会根据边缘每一侧的两个像元点之间的距离进行扩展和收缩，这在voronoi的区域内发生巨大的变化。不管怎样，作为参考，实现应该是这样的：

```c#
vec2 voronoi( in vec2 x )
{
    ivec2 p = floor( x );
    vec2  f = fract( x );

    vec2 res = vec2( 8.0 );
    for( int j=-1; j<=1; j++ )
    for( int i=-1; i<=1; i++ )
    {
        ivec2 b = ivec2(i, j);
        vec2  r = vec2(b) - f + random2f(p + b);
        float d = dot(r, r);

        if( d < res.x )
        {
            res.y = res.x;
            res.x = d;
        }
        else if( d < res.y )
        {
            res.y = d;
        }
    }

    return sqrt( res );
}
float getBorder( in vec2 p )
{
    vec2 c = voronoi( p );

    float dis = c.y - c.x;

    return 1.0 - smoothstep(0.0,0.05,dis);
}
```

==Some ways to almost solve the problem==

​	估计实际距离的一种简单方法是取函数F2-F1，计算它的梯度，然后将F2-F1除以梯度的模。您可以在本文中了解如何做到这一点。这实际上是一个通用的方法，它在大多数情况下都能正常工作(但并非总是如此)。问题是，当然，它非常缓慢，因为它需要三个额外的voronoi评估。我们可以做得更好。

​	好的voronoi实现不仅返回最近点的距离，而且返回点本身（位置和ID）。如果我们知道voronoi网格最近的两个点在哪里，那么我们可能可以更好地近似到分隔细胞的线的距离。

![](https://jmx-paper.oss-cn-beijing.aliyuncs.com/IQ%E5%A4%A7%E7%A5%9E%E5%8D%9A%E5%AE%A2%E9%98%85%E8%AF%BB/%E5%9B%BE%E7%89%87/procedural%20content/VE1.jpg)

​	当然，该线是将两点*a*和*b*之间的红色线段一分为二的线。它通过点*m*，该点恰好是*a*和*b*的平均值。蓝点*x*是我们要着色的点，到边界的距离是紫色线段的长度。因此，我们只需要沿着*b-a*的方向投影向量*x-m*，这为我们提供了紫色矢量的长度。
$$
distance=<x-\frac{a+b}{2},\frac{b-a}{|b-a|}>
$$

```c#
vec2 voronoi( in vec2 x, out vec2 oA, out vec2 oB )
{
    ivec2 p = floor( x );
    vec2  f = fract( x );

    vec2 res = vec2( 8.0 );
    for( int j=-1; j<=1; j++ )
    for( int i=-1; i<=1; i++ )
    {
        ivec2 b = ivec2(i, j);
        vec2  r = vec2(b) - f + random2f(p+b);
        float d = dot( r, r );

        if( d < res.x )
        {
            res.y = res.x;
            res.x = d;
            oA = r;
        }
        else if( d < res.y )
        {
            res.y = d;
            oB = r;
        }
    }

    return sqrt( res );
}
float getBorder( in vec2 p )
{
    vec2 a, b;
    vec2 c = voronoi( p, a, b );

    float d = dot(0.5*(a+b),normalize(b-a));

    return 1.0 - smoothstep(0.0,0.05,d);
}
```

==The final algorithm==

​	那么，解决方案必须是首先检测哪个单元格包含最接近着色点x的点，然后以该单元格为中心进行邻居搜索。

```c#
float voronoiDistance( in vec2 x )
{
    ivec2 p = ivec2(floor( x ));
    vec2  f = fract( x );

    ivec2 mb;
    vec2 mr;

    float res = 8.0;
    for( int j=-1; j<=1; j++ )
    for( int i=-1; i<=1; i++ )
    {
        ivec2 b = ivec2(i, j);
        vec2  r = vec2(b) + random2f(p+b)-f;
        float d = dot(r,r);

        if( d < res )
        {
            res = d;
            mr = r;
            mb = b;
        }
    }

    res = 8.0;
    for( int j=-2; j<=2; j++ )
    for( int i=-2; i<=2; i++ )
    {
        ivec2 b = mb + ivec2(i, j);
        vec2  r = vec2(b) + random2f(p+b) - f;
        float d = dot(0.5*(mr+r), normalize(r-mr));

        res = min( res, d );
    }

    return res;
}
float getBorder( in vec2 p )
{
    float d = voronoiDistance( p );

    return 1.0 - smoothstep(0.0,0.05,d);
}
```

==亮点技巧==

产生等高线的[简单公式](https://www.shadertoy.com/view/ldl3W8)

```
vec3 col = w*(0.5 + 0.5*sin(64.0*w))*vec3(1.0);
```

![](https://jmx-paper.oss-cn-beijing.aliyuncs.com/IQ%E5%A4%A7%E7%A5%9E%E5%8D%9A%E5%AE%A2%E9%98%85%E8%AF%BB/%E5%9B%BE%E7%89%87/procedural%20content/VE2.PNG)





## Smooth Voronoi

​	Voronoi图案在计算机图形学中被广泛用于程序化建模和着色/贴图。然而，当用于着色时，必须格外小心，因为voronoi信号的定义是不连续的，因此很难过滤。这就是为什么通常情况下，这些图案会被超采样并被烘焙成纹理的原因。让我们看看能不能从源头上解决这个丑陋的不连续性。

![](https://jmx-paper.oss-cn-beijing.aliyuncs.com/IQ%E5%A4%A7%E7%A5%9E%E5%8D%9A%E5%AE%A2%E9%98%85%E8%AF%BB/%E5%9B%BE%E7%89%87/procedural%20content/gfx00.jpg)

​	通常voronoi函数会返回许多信号，比如最近特征的距离、id和位置。但是，为了保持简单，这次让我们编写一个非常简单和经典的voronoi模式实现

```c#
float voronoi( in vec2 x )
{
    ivec2 p = floor( x );
    vec2  f = fract( x );

    float res = 8.0;
    for( int j=-1; j<=1; j++ )
    for( int i=-1; i<=1; i++ )
    {
        ivec2 b = ivec2( i, j );
        vec2  r = vec2( b ) - f + random2f( p + b );
        float d = dot( r, r );

        res = min( res, d );
    }
    return sqrt( res );
}
```

​	正如预期的那样，将域划分为一个网格，确定当前阴影点x所在的单元格，扫描以其为中心的3x3单元格，在这9个单元格中的每一个单元格中随机生成一个点，并记录从x到最近的点的距离。到目前为止还算不错。

​	回到我们的不连续性问题。voronoi模式的问题当然是min()操作，这就是不连续的地方。因此，今天的想法是用一个足够相似，但是连续的东西来代替那个min()运算符。如果你仔细想一想，这里的概念是，我们有一组9个点，我们要从中只选取一个--最接近的一个。那么，如果我们不是只挑一个点，而是把它们都挑出来，而是给最接近的一个点以最大的相关性，但不是完全的重要性，这样，当我们在这个域中移动，新的点越来越接近我们的阴影点时，我们就可以顺利地把重要性从最接近的旧点转移到新的点上，会怎样呢？换句话说，如果我们不选择最近的点的距离，而是对所有点的距离进行加权平均，这样我们就可以保留了众所周知的voronoi外观。

​	当然，这可以用很多方法来实现。例如，我们可以用距离的倒数作为近似度系数，然后把它们加起来，最后再把倒数和幂加起来，希望最接近的距离比其他的距离更接近。这个方法效果相当好，但可能会有精度问题。

​	另一种方法是使用快速衰减的指数距离。这样做的效果更好，并且提供了一个非常直观的平滑度控制，缺点是它需要在内循环中多一个平方根。

```c#
float smoothVoronoi( in vec2 x )
{
    ivec2 p = floor( x );
    vec2  f = fract( x );

    float res = 0.0;
    for( int j=-1; j<=1; j++ )
    for( int i=-1; i<=1; i++ )
    {
        ivec2 b = ivec2( i, j );
        vec2  r = vec2( b ) - f + random2f( p + b );
        float d = dot( r, r );

        res += 1.0/pow( d, 8.0 );
    }
    return pow( 1.0/res, 1.0/16.0 );
}

float smoothVoronoi( in vec2 x )
{
    ivec2 p = floor( x );
    vec2  f = fract( x );

    float res = 0.0;
    for( int j=-1; j<=1; j++ )
    for( int i=-1; i<=1; i++ )
    {
        ivec2 b = ivec2( i, j );
        vec2  r = vec2( b ) - f + random2f( p + b );
        float d = dot( r, r );

        res += 1.0/pow( d, 8.0 );
    }
    return pow( 1.0/res, 1.0/16.0 );
}
float smoothVoronoi( in vec2 x )
{
    ivec2 p = floor( x );
    vec2  f = fract( x );

    float res = 0.0;
    for( int j=-1; j<=1; j++ )
    for( int i=-1; i<=1; i++ )
    {
        ivec2 b = ivec2( i, j );
        vec2  r = vec2( b ) - f + random2f( p + b );
        float d = length( r );

        res += exp( -32.0*d );
    }
    return -(1.0/32.0)*log( res );
}
```

<details>    
<summary>戳纸代码</summary>    
<pre><code>  
vec2 hash22(vec2 p)
{
	vec3 p3 = fract(vec3(p.xyx) * vec3(.1031, .1030, .0973));
    p3 += dot(p3, p3.yzx+33.33);
    return fract((p3.xx+p3.yz)*p3.zy);
}
float voronoi(in vec2 x)
{
    vec2 p=floor(x);
    vec2 f=fract(x);
    float res=8.0;
    for(int j=-2;j<=2;j++)
        for(int i=-2;i<=2;i++)
        {
            vec2 b=vec2(i,j);
            vec2 o=hash22(p+b);
            //o=o*0.5+0.5*sin(iTime+o*64.);
            vec2 r=b+o-f;
            float d = length( r );
            res += exp( -2.0*d );
        }
    res =-(1.0/2.0)*log( res );
    return 1.-res;
}
vec3 getNormal(in vec2 p)
{
    float eps=0.0001;
    vec2 h=vec2(eps,0.);
    return normalize(vec3(voronoi(p-h.xy)-voronoi(p+h.xy),2.0*h.x,voronoi(p-h.yx)-voronoi(p+h.yx)));
}
void mainImage( out vec4 fragColor, in vec2 fragCoord )
{
    vec2 p=(fragCoord*2.-iResolution.xy)/iResolution.y;
    vec3 nor=getNormal(p*3.);
    vec3 mate=vec3(0.4);
    vec3 lig1=normalize(vec3(1.0,1.0,1.0));
    float dif1=clamp(dot(nor,lig1),0.0,1.0);
    vec3 col=mate*4.*dif1*vec3(0.7,0.75,0.7);
    fragColor=vec4(col,1.);
}
</code></pre>
</details>

![](https://jmx-paper.oss-cn-beijing.aliyuncs.com/IQ%E5%A4%A7%E7%A5%9E%E5%8D%9A%E5%AE%A2%E9%98%85%E8%AF%BB/%E5%9B%BE%E7%89%87/procedural%20content/SVE.PNG)

![](https://jmx-paper.oss-cn-beijing.aliyuncs.com/IQ%E5%A4%A7%E7%A5%9E%E5%8D%9A%E5%AE%A2%E9%98%85%E8%AF%BB/%E5%9B%BE%E7%89%87/procedural%20content/SVE2.PNG)







## Voronoise

​	程序模式生成的两个最常见的构建块：噪声，有许多变化（Perlin是第一个也是最相关的）和Voronoi（也称为“celular”），也有不同的变化。对于Voronoi来说，这些变化中最常见的是在一个规则网格中分割域，这样每个单元中都有一个特征点。这意味着Voronoi模式是基于网格的，就像噪音一样，不同之处在于，在噪音中，特征发起者位于网格的顶点(随机值或随机梯度)，而Voronoi的特征生成器则在网格的某个地方抖动。

​	尽管存在相似之处，但事实是两种模式中使用网格的方式都不同。噪声对随机值（如[值noise](http://en.wikipedia.org/wiki/Value_noise)）或梯度（如[梯度noise](http://en.wikipedia.org/wiki/Gradient_noise)）进行插值/平均，而Voronoi计算到最近特征点的距离。平滑双线性插值和最小评估是两个非常不同的运算。它们是否可以结合在一个更一般的度量标准中呢?

​	本文是关于寻找这种通用模式的一个小尝试。当然，实现这种泛化的代码永远不会像特定情况下的实现那样快（这篇文章没有明显的实际用途），但至少它可能为更全面的理解打开了一扇窗，也许有一天，还会有新的发现。

==The Code==

​	为了推广Voronoi和噪声，我们必须引入两个参数：一个用于控制特征点的抖动量，另一个用于控制度量（metric）。我们称网格控制参数为u，称度量（metric）控制器为v。

​	grid参数的设计非常简单：**u = 0**将仅使用类似于噪声的常规网格，而**u = 1**将是类似于Voronoi的抖动网格。因此，**u**的值可以简单地控制抖动量。

​	该**v**参数必须混合在噪声的双线性内插值和Voronoi的min运算符之间。这里的主要困难是min操作是一个非连续函数。但是，对我们来说幸运的是，还有一些平滑的替代方法，例如[Smooth Voronoi ](https://www.iquilezles.org/www/articles/smoothvoronoi/smoothvoronoi.htm)。如果我们对每个特征点的距离应用幂函数以突出显示其余特征上最接近的那个，那么我们会得到一个不错的副作用：使用1的幂使所有特征具有相同的相关性，因此我们得到了相等的插值的功能，这就是我们需要的噪声模式！因此，可能会执行以下操作：

```c#
float ww=pow(1.0-smoothstep(0.0,1.414,sqrt(d)),64.-63.*v);
```

一些实验证明，通过将**v增大**至某个幂，可以在类噪声模式和类Voronoi模式之间实现更好的感知线性插值：

```c#
float ww = pow(1.0-smoothstep(0.0,1.414,sqrt(d)),1.0 + 63.0 * pow(1.0-v，4.0));
```

然后，我们的新通用超模式的代码可以是这样的

```c#
float noise( in vec2 x, float u, float v )
{
    vec2 p = floor(x);
    vec2 f = fract(x);

    float k = 1.0 + 63.0*pow(1.0-v,4.0);
    float va = 0.0;
    float wt = 0.0;
    for( int j=-2; j<=2; j++ )
    for( int i=-2; i<=2; i++ )
    {
        vec2  g = vec2( float(i), float(j) );
        vec3  o = hash3( p + g )*vec3(u,u,1.0);
        vec2  r = g - f + o.xy;
        float d = dot(r,r);
        float w = pow( 1.0-smoothstep(0.0,1.414,sqrt(d)), k );
        va += w*o.z;
        wt += w;
    }

    return va/wt;
}
```

==The Result==

[例子](https://www.shadertoy.com/view/Xd23Dh)

![](https://jmx-paper.oss-cn-beijing.aliyuncs.com/IQ%E5%A4%A7%E7%A5%9E%E5%8D%9A%E5%AE%A2%E9%98%85%E8%AF%BB/%E5%9B%BE%E7%89%87/procedural%20content/VN1.jpg)

​				左下角：u = 0，v = 0：单元噪声
​				右下角：u = 0，v = 1：噪声
​				左上角：u = 1，v = 0：Voronoi
​				右上角：u = 1，v = 1：Voronoise







## Advanced Value Noise

在这里，我将写一些关于值噪声的有趣的事实（与梯度噪声相似，但不相同，其中Perlin噪声是一个可能的实现）。是的，令人难以置信的是，你会在这里找到一些你在其他地方找不到的信息，并没有多少人谈论值噪声的导数，比如如何分析计算它们以及如何处理它们。那么，为什么不在这里做呢?它们毕竟是非常有用的。

==The derivatives==

我们称我们的3d值噪声为n(x,y,z)，任何维度都是一样的。(3d)噪声函数它是基于在给定纬度点上随机值的(tri)线性插值。像这样的东西

```c#
n = lerp(w, lerp(v, lerp(u, a, b) ,lerp(u, c, d)),lerp(v, lerp(u, e, f), lerp(u, g, h)));
```

其中u(x) v(y) w(z)通常是这种形式的三次或五次多项式：
$$
\begin{align}
u(x)&=3x^2-2x^3\\
u(x)&=6x^5-15x^4+10x^3
\end{align}
$$
现在，n(x,y,z)可以被拓展成
$$
n(u,v,w)=k_0+k_1\cdot u+k_2\cdot v+k_3\cdot w+k_4\cdot uv+k_5\cdot vw+k_6\cdot wu+k_7\cdot uvw
$$
with
$$
\begin{align}
k_0&=a\\
k_1&=b-a\\
k_2&=c-a\\
k_3&=e-a\\
k_4&=a-b-c+d\\
k_5&=a-c-e+g\\
k_6&=a-b-e+f\\
k_7&=-a+b+c-d+e-f-g+h
\end{align}
$$
导数现在可以很容易地计算出来，例如，对于x：
$$
\partial n/\partial x=(k_1+k_4\cdot v+k_6\cdot w+k_7\cdot vw)\cdot u^`(x)
$$
with
$$
u^`(x)=6\cdot x\cdot(1-x)
$$

```c#
// returns 3D value noise and its 3 derivatives
 vec4 noised( in vec3 x )
 {
    vec3 p = floor(x);
    vec3 w = fract(x);

    vec3 u = w*w*w*(w*(w*6.0-15.0)+10.0);
    vec3 du = 30.0*w*w*(w*(w-2.0)+1.0);

    float a = myRandomMagic( p+vec3(0,0,0) );
    float b = myRandomMagic( p+vec3(1,0,0) );
    float c = myRandomMagic( p+vec3(0,1,0) );
    float d = myRandomMagic( p+vec3(1,1,0) );
    float e = myRandomMagic( p+vec3(0,0,1) );
    float f = myRandomMagic( p+vec3(1,0,1) );
    float g = myRandomMagic( p+vec3(0,1,1) );
    float h = myRandomMagic( p+vec3(1,1,1) );

    float k0 =   a;
    float k1 =   b - a;
    float k2 =   c - a;
    float k3 =   e - a;
    float k4 =   a - b - c + d;
    float k5 =   a - c - e + g;
    float k6 =   a - b - e + f;
    float k7 = - a + b + c - d + e - f - g + h;

    return vec4( -1.0+2.0*(k0 + k1*u.x + k2*u.y + k3*u.z + k4*u.x*u.y + k5*u.y*u.z + k6*u.z*u.x + k7*u.x*u.y*u.z),
                 2.0* du * vec3( k1 + k4*u.y + k6*u.z + k7*u.y*u.z,
                                 k2 + k5*u.z + k4*u.x + k7*u.z*u.x,
                                 k3 + k6*u.x + k5*u.y + k7*u.x*u.y ) );
}
```

==FBM derivatives==

fBM（[分数布朗运动](https://www.iquilezles.org/www/articles/fbm/fbm.htm)）通常被实现为Value Noise函数的分数和：

![](https://jmx-paper.oss-cn-beijing.aliyuncs.com/IQ%E5%A4%A7%E7%A5%9E%E5%8D%9A%E5%AE%A2%E9%98%85%E8%AF%BB/%E5%9B%BE%E7%89%87/procedural%20content/gfx_f10.png)

> with w=1/2 and s=2, or something close, normally. When s=2 each iteration is called a "octave" - for the doubling of the frequency, like in music. The total derivative is in that case the weighted sum of the derivatives for each octave of course, as in regular derivative rules. If you implement a ridged Value Noise or other variations you can also easily drive the right way to combine the derivatives, unless you have a discontinuous shaping function like a fabsf().

```c#
// returns 3D fbm and its 3 derivatives
vec4 fbm( in vec3 x, int octaves )
{
    float f = 1.98;  // could be 2.0
    float s = 0.49;  // could be 0.5
    float a = 0.0;
    float b = 0.5;
    vec3  d = vec3(0.0);
    mat3  m = mat3(1.0,0.0,0.0,
    0.0,1.0,0.0,
    0.0,0.0,1.0);
    for( int i=0; i < octaves; i++ )
    {
        vec4 n = noised(x);
        a += b*n.x;          // accumulate values
        d += b*m*n.yzw;      // accumulate derivatives
        b *= s;
        x = f*m3*x;
        m = f*m3i*m;
    }
    return vec4( a, d );
}
```

==Other Use==

噪声导数的另一种用法是修改fbm（）构造以获得不同的外观。例如，将衍生物注入fbm的核心就足够了，可以模拟不同的腐蚀样效果，并为地形创建一些丰富多样的形状，包括平坦区域和更粗糙的区域。

![](https://jmx-paper.oss-cn-beijing.aliyuncs.com/IQ%E5%A4%A7%E7%A5%9E%E5%8D%9A%E5%AE%A2%E9%98%85%E8%AF%BB/%E5%9B%BE%E7%89%87/procedural%20content/AVN1.PNG)

```c#
const mat2 m = mat2(0.8,-0.6,0.6,0.8);

float terrain( in vec2 p )
{
    float a = 0.0;
    float b = 1.0;
    vec2  d = vec2(0.0);
    for( int i=0; i<15; i++ )
    {
        vec3 n=noised(p);
        d +=n.yz;
        a +=b*n.x/(1.0+dot(d,d));
        b *=0.5;
        p=m*p*2.0;
    }
    return a;
    }
```







## Gradient Noise Derivatives

与[值噪声](https://www.iquilezles.org/www/articles/morenoise/morenoise.htm)的[分析导数](https://www.iquilezles.org/www/articles/morenoise/morenoise.htm)类似，梯度噪声（用于Perlin噪声的变化和概括的名称）接受对导数的解析计算。就像值噪声导数一样，这允许更快的光照计算或任何其他基于噪声的梯度/法线的计算，因为我们不再需要通过涉及多个噪声样本的数值方法来近似它。[例子](https://www.shadertoy.com/view/4dffRH)

```c#
// returns 3D value noise
float noise( in vec3 x )
{
    // grid
    vec3 p = floor(x);
    vec3 w = fract(x);
    
    // quintic interpolant
    vec3 u = w*w*w*(w*(w*6.0-15.0)+10.0);

    
    // gradients
    vec3 ga = hash( p+vec3(0.0,0.0,0.0) );
    vec3 gb = hash( p+vec3(1.0,0.0,0.0) );
    vec3 gc = hash( p+vec3(0.0,1.0,0.0) );
    vec3 gd = hash( p+vec3(1.0,1.0,0.0) );
    vec3 ge = hash( p+vec3(0.0,0.0,1.0) );
    vec3 gf = hash( p+vec3(1.0,0.0,1.0) );
    vec3 gg = hash( p+vec3(0.0,1.0,1.0) );
    vec3 gh = hash( p+vec3(1.0,1.0,1.0) );
    
    // projections
    float va = dot( ga, w-vec3(0.0,0.0,0.0) );
    float vb = dot( gb, w-vec3(1.0,0.0,0.0) );
    float vc = dot( gc, w-vec3(0.0,1.0,0.0) );
    float vd = dot( gd, w-vec3(1.0,1.0,0.0) );
    float ve = dot( ge, w-vec3(0.0,0.0,1.0) );
    float vf = dot( gf, w-vec3(1.0,0.0,1.0) );
    float vg = dot( gg, w-vec3(0.0,1.0,1.0) );
    float vh = dot( gh, w-vec3(1.0,1.0,1.0) );
	
    // interpolation
    return va + 
           u.x*(vb-va) + 
           u.y*(vc-va) + 
           u.z*(ve-va) + 
           u.x*u.y*(va-vb-vc+vd) + 
           u.y*u.z*(va-vc-ve+vg) + 
           u.z*u.x*(va-vb-ve+vf) + 
           u.x*u.y*u.z*(-va+vb+vc-vd+ve-vf-vg+vh);
}
```

```c#
// returns 3D value noise (in .x)  and its derivatives (in .yzw)
vec4 noised( in vec3 x )
{
    // grid
    vec3 p = floor(x);
    vec3 w = fract(x);
    
    // quintic interpolant
    vec3 u = w*w*w*(w*(w*6.0-15.0)+10.0);
    vec3 du = 30.0*w*w*(w*(w-2.0)+1.0);
    
    // gradients
    vec3 ga = hash( p+vec3(0.0,0.0,0.0) );
    vec3 gb = hash( p+vec3(1.0,0.0,0.0) );
    vec3 gc = hash( p+vec3(0.0,1.0,0.0) );
    vec3 gd = hash( p+vec3(1.0,1.0,0.0) );
    vec3 ge = hash( p+vec3(0.0,0.0,1.0) );
    vec3 gf = hash( p+vec3(1.0,0.0,1.0) );
    vec3 gg = hash( p+vec3(0.0,1.0,1.0) );
    vec3 gh = hash( p+vec3(1.0,1.0,1.0) );
    
    // projections
    float va = dot( ga, w-vec3(0.0,0.0,0.0) );
    float vb = dot( gb, w-vec3(1.0,0.0,0.0) );
    float vc = dot( gc, w-vec3(0.0,1.0,0.0) );
    float vd = dot( gd, w-vec3(1.0,1.0,0.0) );
    float ve = dot( ge, w-vec3(0.0,0.0,1.0) );
    float vf = dot( gf, w-vec3(1.0,0.0,1.0) );
    float vg = dot( gg, w-vec3(0.0,1.0,1.0) );
    float vh = dot( gh, w-vec3(1.0,1.0,1.0) );
	
    // interpolation
    float v = va + 
              u.x*(vb-va) + 
              u.y*(vc-va) + 
              u.z*(ve-va) + 
              u.x*u.y*(va-vb-vc+vd) + 
              u.y*u.z*(va-vc-ve+vg) + 
              u.z*u.x*(va-vb-ve+vf) + 
              u.x*u.y*u.z*(-va+vb+vc-vd+ve-vf-vg+vh);
              
    vec3 d = ga + 
             u.x*(gb-ga) + 
             u.y*(gc-ga) + 
             u.z*(ge-ga) + 
             u.x*u.y*(ga-gb-gc+gd) + 
             u.y*u.z*(ga-gc-ge+gg) + 
             u.z*u.x*(ga-gb-ge+gf) + 
             u.x*u.y*u.z*(-ga+gb+gc-gd+ge-gf-gg+gh) +   
             
             du * (vec3(vb-va,vc-va,ve-va) + 
                   u.yzx*vec3(va-vb-vc+vd,va-vc-ve+vg,va-vb-ve+vf) + 
                   u.zxy*vec3(va-vb-ve+vf,va-vb-vc+vd,va-vc-ve+vg) + 
                   u.yzx*u.zxy*(-va+vb+vc-vd+ve-vf-vg+vh) ));
                   
    return vec4( v, d );                   
}
```

如果是[2D](https://www.shadertoy.com/view/XdXBRH)，代码自然会变小：

```c#
// returns 3D value noise (in .x)  and its derivatives (in .yz)
vec3 noised( in vec2 x )
{
    vec2 i = floor( p );
    vec2 f = fract( p );

    vec2 u = f*f*f*(f*(f*6.0-15.0)+10.0);
    vec2 du = 30.0*f*f*(f*(f-2.0)+1.0);
    
    vec2 ga = hash( i + vec2(0.0,0.0) );
    vec2 gb = hash( i + vec2(1.0,0.0) );
    vec2 gc = hash( i + vec2(0.0,1.0) );
    vec2 gd = hash( i + vec2(1.0,1.0) );
    
    float va = dot( ga, f - vec2(0.0,0.0) );
    float vb = dot( gb, f - vec2(1.0,0.0) );
    float vc = dot( gc, f - vec2(0.0,1.0) );
    float vd = dot( gd, f - vec2(1.0,1.0) );

    return vec3( va + u.x*(vb-va) + u.y*(vc-va) + u.x*u.y*(va-vb-vc+vd),   // value
                 ga + u.x*(gb-ga) + u.y*(gc-ga) + u.x*u.y*(ga-gb-gc+gd) +  // derivatives
                 du * (u.yx*(va-vb-vc+vd) + vec2(vb,vc) - va));
}
```





## Filtering The Checkerboard Pattern

棋盘图案通常在计算机图形工具和纸张中被视为更复杂的图案或纹理的占位符。尽管如此，当使用它时，仍然没有理由草率-用户仍然期望图像看起来高质量，自然包括适当的抗锯齿。一种方法是将实际的棋盘格图案简单地存储在mipmapped纹理中，在这种情况下，将以通常的方式执行过滤。但是，由于不同的原因，您可能要避免使用纹理作为图案，而是按程序进行。然后，本文可能会引起您的兴趣，因为它说明了如何分析性地执行棋盘图案的过滤（没有预先计算的纹理或mips）