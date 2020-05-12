# IQ大神博客阅读心得7

| 名称                              | 概述                         |
| --------------------------------- | ---------------------------- |
| [Popcorn Images](#Popcorn-Images) | Pop图片的生成方法            |
| [Domain Warping](#Domain_Warping) | Warping图的生成方法          |
| [Voronoi Edges](#Voronoi-Edges)   | 细胞图的生成和边界问题       |
| [Smooth Voronoi](#Smooth-Voronoi) | 细胞图不连续性的几个解决方法 |
| [Voronoise](#Voronoise)           |                              |
|                                   |                              |
|                                   |                              |
|                                   |                              |
|                                   |                              |
|                                   |                              |
|                                   |                              |







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

任何使用voronoi序列创造爬行动物皮肤纹理或干地砖的人都知道，当使用F2 - F1并对其进行阈值化时，分隔细胞的线的宽度是不一致的。这是一个恼人的问题。有技术可以缓解这个问题，但它们只是近似的。在这里，我们将找到voronoi噪声的实现，它可以生成数学上完美的细胞分离线，并且是完全程序化的。

==The Problem==

从距离第二个最近点的距离中减去到最近点的距离，或者人们所说的*F2-F1 Voronoi*，它非常接近单元格边界生成器。实际上，单元格的边界发生在这两个距离相等的位置（两个最近邻居的等距点），因此函数F2-F1恰好在单元格的边界取值0.0，这非常有用。因此，一个人很容易在两个小数字之间简单地平滑F2-F1，并将其称为“单元边缘”。这种工作，但不是完全。F2-F1并不是真正的距离，因为它会根据边缘每一侧的两个像元点之间的距离进行扩展和收缩，这在voronoi的区域内发生巨大的变化。不管怎样，作为参考，实现应该是这样的：

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

估计实际距离的一种简单方法是取函数F2-F1，计算它的梯度，然后将F2-F1除以梯度的模。您可以在本文中了解如何做到这一点。这实际上是一个通用的方法，它在大多数情况下都能正常工作(但并非总是如此)。问题是，当然，它非常缓慢，因为它需要三个额外的voronoi评估。我们可以做得更好。

好的voronoi实现不仅返回最近点的距离，而且返回点本身（位置和ID）。如果我们知道voronoi网格最近的两个点在哪里，那么我们可能可以更好地近似到分隔细胞的线的距离。

![](https://jmx-paper.oss-cn-beijing.aliyuncs.com/IQ%E5%A4%A7%E7%A5%9E%E5%8D%9A%E5%AE%A2%E9%98%85%E8%AF%BB/%E5%9B%BE%E7%89%87/procedural%20content/VE1.jpg)

当然，该线是将两点*a*和*b*之间的红色线段一分为二的线。它通过点*m*，该点恰好是*a*和*b*的平均值。蓝点*x*是我们要着色的点，到边界的距离是紫色线段的长度。因此，我们只需要沿着*b-a*的方向投影向量*x-m*，这为我们提供了紫色矢量的长度。
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

那么，解决方案必须是首先检测哪个单元格包含最接近着色点x的点，然后以该单元格为中心进行邻居搜索。

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

Voronoi图案在计算机图形学中被广泛用于程序化建模和着色/贴图。然而，当用于着色时，必须格外小心，因为voronoi信号的定义是不连续的，因此很难过滤。这就是为什么通常情况下，这些图案会被超采样并被烘焙成纹理的原因。让我们看看能不能从源头上解决这个丑陋的不连续性。

![](https://jmx-paper.oss-cn-beijing.aliyuncs.com/IQ%E5%A4%A7%E7%A5%9E%E5%8D%9A%E5%AE%A2%E9%98%85%E8%AF%BB/%E5%9B%BE%E7%89%87/procedural%20content/gfx00.jpg)

通常voronoi函数会返回许多信号，比如最近特征的距离、id和位置。但是，为了保持简单，这次让我们编写一个非常简单和经典的voronoi模式实现

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

正如预期的那样，将域划分为一个网格，确定当前阴影点x所在的单元格，扫描以其为中心的3x3单元格，在这9个单元格中的每一个单元格中随机生成一个点，并记录从x到最近的点的距离。到目前为止还算不错。

回到我们的不连续性问题。voronoi模式的问题当然是min()操作，这就是不连续的地方。因此，今天的想法是用一个足够相似，但是连续的东西来代替那个min()运算符。如果你仔细想一想，这里的概念是，我们有一组9个点，我们要从中只选取一个--最接近的一个。那么，如果我们不是只挑一个点，而是把它们都挑出来，而是给最接近的一个点以最大的相关性，但不是完全的重要性，这样，当我们在这个域中移动，新的点越来越接近我们的阴影点时，我们就可以顺利地把重要性从最接近的旧点转移到新的点上，会怎样呢？换句话说，如果我们不选择最近的点的距离，而是对所有点的距离进行加权平均，这样我们就可以保留了众所周知的voronoi外观。

当然，这可以用很多方法来实现。例如，我们可以用距离的倒数作为近似度系数，然后把它们加起来，最后再把倒数和幂加起来，希望最接近的距离比其他的距离更接近。这个方法效果相当好，但可能会有精度问题。

另一种方法是使用快速衰减的指数距离。这样做的效果更好，并且提供了一个非常直观的平滑度控制，缺点是它需要在内循环中多一个平方根。

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

