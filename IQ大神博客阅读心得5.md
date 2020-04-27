## IQ大神博客阅读心得5

| 标题                                                         | 简介                       |
| ------------------------------------------------------------ | -------------------------- |
| [SSAO](#SSAO)                                                | 简单的屏幕空间环境光遮蔽   |
| [**Better Fog**](#Better-Fog)                                | **关于雾的计算的诸多效果** |
| [Penumbra Shadows In Raymarched SDFS](#Penumbra-Shadows-In-Raymarched-SDFS) |                            |
|                                                              |                            |
|                                                              |                            |
|                                                              |                            |
|                                                              |                            |
|                                                              |                            |
|                                                              |                            |
|                                                              |                            |

------



#### SSAO

```c#
uniform vec4 fk3f[32];
uniform vec4 fres;
uniform sampler2D tex0;
uniform sampler2D tex1;

void main(void)
{
    //采样第一次获得的深度图，获得该像素的Z
    vec4 zbu = texture2D( tex0, gl_Color.xy );
    
    //求得该像素点在视点空间的位置
    vec3 ep = zbu.x*gl_TexCoord[0].xyz/gl_TexCoord[0].z;
    
    //采样随机随机法线贴图，获得一个随机干扰量
    vec4 pl = texture2D( tex1, gl_Color.xy*fres.xy );
    //区间重定向
    pl = pl*2.0 - vec4(1.0);

    float bl = 0.0;
    for( int i=0; i<32; i++ )
    {
        //根据随机干扰量和随机值做反射
        vec3 se = ep + rad*reflect(fk3f[i].xyz,pl.xyz);
        //归一化、但是后面那个乘值不太懂，是因为长宽比是4:3吗
        vec2 ss = (se.xy/se.z)*vec2(.75,1.0);
        //区间重定位
        vec2 sn = ss*.5 + vec2(.5);
        //采样
        vec4 sz = texture2D(tex0,sn);
        //根据采样值计算距离，并进行Step限制
        float zd = 50.0*max( se.z-sz.x, 0.0 );
        //计算AO贡献值
        bl += 1.0/(1.0+zd*zd);
   }
   gl_FragColor = vec4(bl/32.0);
}
```





#### Better Fog

雾是计算机图形学中非常流行的元素，因此非常流行，因此实际上我们总是在教科书或教程中对其进行了介绍。但是，这些教科书，教程甚至API只能进行简单的基于距离的颜色融合。

传统上，雾是作为视觉元素引入的，它在图像中给出距离提示。的确，雾很快就能帮助我们了解物体的距离，从而了解物体的尺度以及世界本身。

![](https://jmx-paper.oss-cn-beijing.aliyuncs.com/IQ%E5%A4%A7%E7%A5%9E%E5%8D%9A%E5%AE%A2%E9%98%85%E8%AF%BB/%E5%9B%BE%E7%89%87/lighting/fog1.jpg)

![](https://jmx-paper.oss-cn-beijing.aliyuncs.com/IQ%E5%A4%A7%E7%A5%9E%E5%8D%9A%E5%AE%A2%E9%98%85%E8%AF%BB/%E5%9B%BE%E7%89%87/lighting/fog2.jpg)

```c#
vec3 applyFog( in vec3  rgb,       // original color of the pixel
               in float distance ) // camera to point distance
{
    float fogAmount = 1.0 - exp( -distance*b );
    vec3  fogColor  = vec3(0.5,0.6,0.7);
    return mix( rgb, fogColor, fogAmount );
}

```

但是，我们应该注意，雾还可以提供更多信息。例如，雾的颜色可以告诉我们有关太阳强度的信息。甚至，如果我们使雾的颜色不是恒定的而是取决于方向的，我们可以为图像引入额外的逼真度。例如，当视图矢量与太阳方向对齐时，我们可以将典型的蓝色雾色更改为淡黄色。这给出了非常自然的光散射效果。有人会说这样的效果不应该称为雾而是散射，我同意，但是到最后，人们只需要稍微修改一下雾方程即可完成效果。

![](https://jmx-paper.oss-cn-beijing.aliyuncs.com/IQ%E5%A4%A7%E7%A5%9E%E5%8D%9A%E5%AE%A2%E9%98%85%E8%AF%BB/%E5%9B%BE%E7%89%87/lighting/fog3.jpg)

```c#
vec3 applyFog( in vec3  rgb,      // original color of the pixel
               in float distance, // camera to point distance
               in vec3  rayDir,   // camera to point vector
               in vec3  sunDir )  // sun light direction
{
    float fogAmount = 1.0 - exp( -distance*b );
    float sunAmount = max( dot( rayDir, sunDir ), 0.0 );
    vec3  fogColor  = mix( vec3(0.5,0.6,0.7), // bluish
                           vec3(1.0,0.9,0.7), // yellowish
                           pow(sunAmount,8.0) );
    return mix( rgb, fogColor, fogAmount );
}
```

效果可以更复杂。例如，太阳向量和视点向量之间的点积指数（当然，它控制方向颜色梯度的影响）也可以随距离而变化。如果设置正确，则可以伪造发光/泛光和其他光散射效果，而无需进行任何多遍处理或渲染纹理，而只需对雾化方程进行简单更改即可。颜色也会随高度或您可能想到的任何其他参数而改变。

该技术的另一种变化是将通常的mix（）命令分为两部分：

```c#
finalColor = pixelColor *（1.0-exp（-distance * b））+ fogColor * exp（-distance * b）;
```

现在，根据经典的CG大气散射论文，第一个术语可以解释为由于散射或“消光”引起的光的聚集，而第二个术语可以解释为“散射”。我们注意到，这种表示雾的方式更为有效，因为现在我们可以为消光和散射选择独立的参数***b***。此外，我们不能有一个或两个，而是最多可以有六个不同的系数-消色的rgb通道三个，散乱的rgb彩色版本三个。

```c#
vec3 extColor = vec3( exp(-distance*be.x), exp(-distance*be.y) exp(-distance*be.z) );
vec3 insColor = vec3( exp(-distance*bi.x), exp(-distance*bi.y) exp(-distance*bi.z) );
finalColor = pixelColor*(1.0-extColor) + fogColor*insColor;
```

这种做雾的方式，结合太阳方向着色和其他技巧，可以为您提供功能强大且简单的雾化系统，同时又非常紧凑，快速。它也非常直观，您无需处理Mie和Rayleight光谱常数之类的物理参数，数学和常数。**简单而可控就是胜利**。

***非恒定密度***

原始和简单的雾化公式具有两个参数：颜色和密度（我在上面的着色器代码中将其称为***b***）。同样，我们将其修改为具有非恒定的颜色，我们也可以对其进行修改以使其不具有恒定的密度。

一般来书，海拔越高，大气层密度越小，我们可以用指数对密度变化建模。指数函数的优势是公式的解是解析的。
$$
d(y)=a\cdot b^{-by}
$$
参数 **b**当然控制该密度的下降。现在，当我们的光线穿过摄影机到点的大气时，它穿过大气层时会积累不透明性。明显我们需要对此进行积分。我们射线的定义为：
$$
r(t)=o_y+t\cdot k_y
$$
我们有雾的总量为
$$
D=\int_0^t{d(y(t))}\cdot dt
$$
从而
$$
D=\int_0^t{d(o_y+t\cdot k_y)}\cdot dt=a\cdot e^{-b\cdot o_y}\frac{1-e^{-b\cdot k_y\cdot t}}{b\cdot k_y}
$$
所以我们的非恒定雾效着色器为

```c#
vec3 applyFog( in vec3  rgb,      // original color of the pixel
               in float distance, // camera to point distance
               in vec3  rayOri,   // camera position
               in vec3  rayDir )  // camera to point vector
{
    float fogAmount = c * exp(-rayOri.y*b) * (1.0-exp( -distance*rayDir.y*b ))/rayDir.y;
    vec3  fogColor  = vec3(0.5,0.6,0.7);
    return mix( rgb, fogColor, fogAmount );
}
```

![](https://jmx-paper.oss-cn-beijing.aliyuncs.com/IQ%E5%A4%A7%E7%A5%9E%E5%8D%9A%E5%AE%A2%E9%98%85%E8%AF%BB/%E5%9B%BE%E7%89%87/lighting/fog4.jpg)





#### Penumbra Shadows In Raymarched SDFS

