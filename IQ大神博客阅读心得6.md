## IQ大神博客阅读心得6

| 名称                                                         | 简介                                                   |
| ------------------------------------------------------------ | ------------------------------------------------------ |
| [Filtering Procedural Textures](#Filtering-Procedural-Textures) | 过程纹理的过滤采样的一般方法；除噪                     |
| [Directional Derivative](#Directional-Derivative)            | 方向导数的分析，由此得到的云等场景中光照的加速计算思路 |
| [Sphere Soft Shadow](#Sphere-Soft-Shadow)                    |                                                        |
|                                                              |                                                        |
|                                                              |                                                        |
|                                                              |                                                        |
|                                                              |                                                        |
|                                                              |                                                        |
|                                                              |                                                        |
|                                                              |                                                        |





#### Filtering Procedural Textures

程序纹理/着色是计算机图形学中的强大工具。它对存储的要求很低，而且不具有拼贴性，并且其自然的适应几何形状的能力使它对许多应用程序都非常有吸引力。但是，与基于位图（bitmap）的纹理方法不同，位图通过mipmap过滤可以轻松避免混叠，而程序模式很难抗锯齿，这在某些情况下会破坏设计的重点。

本文介绍了一种实现过程模式的过滤/抗锯齿的简单方法，该方法不需要手动进行细节钳位，也不需要将图案支持到位图纹理中，这基本上是一种蛮力方法。当然，以前已经使用过这种方法，但是它似乎并不流行。但是，我发现它在实践中表现良好，并且在如今复杂的照明模型中，这种过滤方法似乎很有用。

![](https://jmx-paper.oss-cn-beijing.aliyuncs.com/IQ%E5%A4%A7%E7%A5%9E%E5%8D%9A%E5%AE%A2%E9%98%85%E8%AF%BB/%E5%9B%BE%E7%89%87/lighting/PS.PNG)

==The Filtering==

显而易见的解决方案是超采样。唯一需要注意的就是正确地执行它，即自适应——不要过度采样，不要欠采样。幸运的是，这个问题是计算机图形学中的一个老问题，很久以前就为我们解决了：过滤器足迹（*filter footprints*）。这是GPU在着色器中访问纹理像素时，用来选择纹理的正确Mipmap级别的技术。最后，对于给定的图像像素，我们需要知道它覆盖了纹理或图案的多少面积。

当使用位图纹理时，该问题可以表述为“我们的纹理中确实有多少纹理像素落在此像素之下”。实际上必须取所有这些纹理像素并将其平均为一种颜色，因为像素只能存储一种颜色。所谓的mipmap中以不同的纹理像素数预先计算的（平均或积分），

对于没有经过缓存/位图烘焙过程的过程模式，由于我们没有预先计算的纹理像素，因此无法执行这种预先集成/预先计算。因此，我们必须委托集成，直到图案/阴影生成时间（即渲染时间）为止。至于过滤器宽度的计算，在位图文本和过程模式之间不会改变。

因此，让我们先关注过滤，然后再关注过滤器占用空间。假设我们确实有一个名为*sampleTexture（）*的过程模式/纹理，如下所示：

```c#
vec3 sampleTexture( in vec3 uvw );
```

```c#
// sample a procedural pattern with filtering
vec3 sampleTextureWithFilter( in vec3 uvw, in vec3 uvwX, in vec3 uvwY, in float detail )
{
    int sx = 1 + iclamp( int( detail*length(uvwX-uvw) ), 0, MaxSamples-1 );
    int sy = 1 + iclamp( int( detail*length(uvwY-uvw) ), 0, MaxSamples-1 );

    vec3 no = vec3( 0.0f );

    for( int j=0; j < sy; j++ )
    for( int i=0; i < sx; i++ )
    {
        vec2 st = vec2( float(i), float(j) )/vec2(float(sx),float(sy));
        no += sampleTexture( uvw + st.x * (ddx_uvw-uvw) + st.y*(ddy_uvw-uvw) );
    }

    return no / float(sx*sy);
}
```

在调用此函数之前，我们首先对于当前坐标fragCoord在X，Y轴进行偏移

```c#
calcRayForPixel( fragCoord.xy + vec2(1.0,0.0), ddx_ro, ddx_rd );
calcRayForPixel( fragCoord.xy + vec2(0.0,1.0), ddy_ro, ddy_rd );
```

然后在相交测试之后，我们依据上述结果进行射线微分的计算（与切平面相交来计算），关于这个计算，我的理解是，如图，除法的结果就是红色除以绿色，依据这个倍数延长射线，与切线平面相交。

![](https://jmx-paper.oss-cn-beijing.aliyuncs.com/IQ%E5%A4%A7%E7%A5%9E%E5%8D%9A%E5%AE%A2%E9%98%85%E8%AF%BB/%E5%9B%BE%E7%89%87/render%20techniques/%E6%89%8B%E7%BB%98.png)

```c#
vec3 ddx_pos = ddx_ro - ddx_rd*dot(ddx_ro-pos,nor)/dot(ddx_rd,nor);
vec3 ddy_pos = ddy_ro - ddy_rd*dot(ddy_ro-pos,nor)/dot(ddy_rd,nor);
```

然后计算纹理采样足迹

```c#
vec3 texCoords( in vec3 p )
{
	return 64.0*p;
}
vec3     uvw = texCoords(     pos );
vec3 ddx_uvw = texCoords( ddx_pos );
vec3 ddy_uvw = texCoords( ddy_pos );
```

最后调用之前的那个函数。

![](https://jmx-paper.oss-cn-beijing.aliyuncs.com/IQ%E5%A4%A7%E7%A5%9E%E5%8D%9A%E5%AE%A2%E9%98%85%E8%AF%BB/%E5%9B%BE%E7%89%87/render%20techniques/filter.PNG)

[测试案例](https://www.shadertoy.com/view/MdjGR1)





#### Directional Derivative

> [梯度与方向导数](https://www.cnblogs.com/key1994/p/11503840.html)
>
> - 方向导数的本质是一个数值，简单来说其定义为：一个函数沿指定方向的变化率。
> - 梯度与方向导数是有本质区别的，梯度其实是一个向量，其定义为：一个函数对于其自变量分别求偏导数，这些偏导数所组成的向量就是函数的梯度。
> - 梯度垂直于等高线，同时指向高度更高的等高线

对于方向导数
$$
\nabla_vf(x)=\nabla{f(x)}\cdot \frac{v}{|v|}
$$
其中，x是空间内正在渲染的点，f是体积场（volumetric field），然后f(x)将是渲染点的密度（density），$$\nabla f(x)$$是点的梯度（或者说 normal）。如果v是光的方向，那么方程的右边将是常规的兰伯特照明$$N\cdot L$$，而左边可以理解为沿LightView的方向导数，那么，我们可以得到一个优化思路——计算光照时，我们可以直接计算左边的式子，而无需使用Normal，下面是作者的解释

基本上，可以直接在感兴趣的方向上测量变化（导数），而不是在所有可能的方向上提取通用导数。换句话说，我们不需抽取4或6个样本来提取通用导数或梯度，然后将其指向光的方向进行照明，而只需在当前点采样不超过2次的场即可，在与光的方向相距一小段距离的点处（并除以该距离）

```c#
// function : R3->R1 is the volumetric density function
// eps is the diferential unit, based on the current LOD
vec3 calcNormal( in vec3 x, in float eps )
{
    vec2 e = vec2( eps, 0.0 );
    return normalize( vec3( function(x+e.xyy) - function(x-e.xyy),
                            function(x+e.yxy) - function(x-e.yxy),
                            function(x+e.yyx) - function(x-e.yyx) ) );
}

void render( void )
{
    // ...
    float den = function( pos );
    vec3  nor = calcNormal( pos, eps );
    float dif = clamp( dot(nor,light), 0.0, 1.0 );
    // ...
}
```

加速为

```c#
// function : R3->R1 is the volumetric density function
// eps is the diferential unit, based on the current LOD
void render( void )
{
    // ...
    float den = function( pos );
    //下面计算了方向导数
    float dif = clamp( (function(pos+eps*light)-den)/eps, 0.0, 1.0 );
    // ...
}
```

当然，缺点是这仅对少量光源有好处。通常，将需要2个用于云的光源（太阳和天顶）。如果有3个或更多的光源，那么基于渐变的传统照明会更加高效。





#### Sphere Soft Shadow

对于给定的阴影点，空间球体和定向光源（不是区域光），请查看从相关点**ro**沿光方向**rd**传播的光线是否撞击或错过了球体，以及是否错过了多少。光线越接近球体，阴影（半影）越暗。但是，有一个观察：最接近的点与接收点**ro**的距离越远，阴影的强度就越小。换句话说，在这个简化模型中，阴影的暗度取决于两个参数：d和t（如下图），则柔和阴影将与它们的比率**d / t**成正比。

![](https://jmx-paper.oss-cn-beijing.aliyuncs.com/IQ%E5%A4%A7%E7%A5%9E%E5%8D%9A%E5%AE%A2%E9%98%85%E8%AF%BB/%E5%9B%BE%E7%89%87/render%20techniques/gfx01.jpg)

```c#
float sphSoftShadow( in vec3 ro, in vec3 rd, in vec4 sph, in float k )
	{
		vec3 oc = ro - sph.xyz;
		float b = dot( oc, rd );
		float c = dot( oc, oc ) - sph.w*sph.w;
		float h = b*b - c;
		
		float d = -sph.w + sqrt( max(0.0,sph.w*sph.w-h));
		float t = -b     - sqrt( max(0.0,h) );
		return (t<0.0) ? 1.0 : smoothstep( 0.0, 1.0, k*d/t );
	}
```

参数**k**控制阴影半影的清晰度。较高的值使其更清晰。这里的smoothstep（）函数只是为了平滑然后在光和影之间过渡。

上面的代码更快的一种方法是删除平方根。我创建了替代近似值，在该近似值以下会生成物理上不正确的阴影，但仍然合理，因为阴影的清晰度取决于生成阴影的对象与接收阴影的对象之间的距离

```c#
float sphSoftShadow( in vec3 ro, in vec3 rd, in vec4 sph, in float k )
	{
		vec3 oc = ro - sph.xyz;
		float b = dot( oc, rd );
		float c = dot( oc, oc ) - sph.w*sph.w;
		float h = b*b - c;
		
		return (b>0.0) ? step(-0.0001,c) : smoothstep( 0.0, 1.0, h*k/b );
	}

```

