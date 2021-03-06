# IQ分形PDF翻译



## 1. 介绍

​		这项工作将描述Mandelbrot集的一些最重要的特征。 这样的描述将主要包括对集合背后的动力学的分析，我们将从中得出其几何形状的一部分。 “动力学到几何学”这一步骤的名称是很自然的，就像我们正在研究的数学对象一样，这在数学对象中是很自然的，因为其几何形状的定义是根据离散非线性动力学系统给出的。 像几乎所有具有这些特征的系统一样，这样的系统在某些区域表现混乱，导致分形结构的几何形状。

​		像几乎所有关于该主题的文章一样，本研究是基于对研究组的纯实验观察。众所周知，即使在今天，描述分形几何学和混沌理论的唯一方法（或至少是最好的方法）还是基于观察。就像詹姆斯·克莱克在他的《混沌》一书中很好地描述的那样，曼德尔布罗特集不允许使用捷径，这与传统的几何图形不同，...知道哪种图形对应于给定方程的唯一方法是通过反复试验来进行。。但是，以天真大胆的特点为例，在这项工作中，我们将尝试找到一些不存在的快捷方式，以便对Mandelbrot集进行更传统的描述。当然，不是出于这个原因，我们将不再害怕伟大的怪物，它以其无限的美丽继续是最能驯服数学家的动物。

​		因此，作为该工作支持的数学演绎工作的序言，第二章包括实验数学练习，这将为我们将在后续各章中正式论证的概念和思想敞开大门。 这些相同的理论概念将使我们发现倒数第二章有关Mandelbrot集的新难题。





## 2. MANDELBROT集的特征

​		在本章的前两节中，将对曼德尔布洛特作为一个整体进行非常简短的描述。更多信息，请参考书目。从第三节开始，我们将从研究工作本身入手，只以定性的方式解释后续章节中使用的基本概念。

### 2.1 茱莉亚集

​		我们知道，曼德尔布罗特合集诞生于1979年，这要归功于已经很有名的波兰数学家法图。朱莉娅在20世纪第二个十年中研究了朱莉娅合集，虽然合集本身已经成为今天研究的对象。
$$
Z \rightarrow Z^2+C
$$
​		定义在复杂的Z面中，由一个动态系统组成，随着应用的迭代，将Z面的每一个起始点转化为其他相继的点。==有的点远离坐标中心向无穷大的方向移动，而有的点则开始周期性地或混沌地或多或少地在坐标中心附近震荡。这两种态度定义了平面中的周期点区域和分歧点区域。这两类区域之间的边界构成了Julia集，并强烈依赖于常数C==。茱莉亚集呈现出一种非同寻常的几何复杂度，除了C=0的集是单位圆外，其他图形就如下图一样漂亮。

<img src="https://jmx-paper.oss-cn-beijing.aliyuncs.com/IQ%E5%A4%A7%E7%A5%9E%E5%8D%9A%E5%AE%A2%E9%98%85%E8%AF%BB/%E5%9B%BE%E7%89%87/IQ%E5%88%86%E6%9E%90PDF%E7%BF%BB%E8%AF%91/2-1.PNG" style="zoom:50%;" />

​		为了对各种可能的Julia集进行分类，可以采用以下方法：==Julia集可分为两类，第一类由那些Julia集组成——属于单件。 另一个由无穷无尽的孤岛（或Cantor云）组成==。 在图2.1中的四组中，尽管右上角是如此脆弱，以至于将要分解成无限的碎片和形式，但只有右下角属于第二类，其余的属于第一类

​		可以证明，只需在点0（迭代的关键点）处迭代$$Z\rightarrow Z^2 + C$$，通过查看该点是否逃脱到无穷大，即可轻松知道属于哪一类。 在这种情况下，Julia集将脱节，而其他情况下则将牢固地结合在一起。

### 2.2 曼德布罗集

​		Mandelbrot决定在C平面上标记出相关的Julia集的起源点。对于平面的每一个C，他计算出初始值z0=0的朱利亚集定义的迭代，并将那些没有逃逸到无穷大的点标记出来。这类点的集合组成了曼德尔布罗特集。其正式定义是：
$$
M(C)=\{c\in C|\{ f_c^n(0)\}^\infty_{n=1}\neq \infty \}
$$
曼德布洛集，根据这个定义，很像茱莉亚集合，如下图2-2所示。

<img src="https://jmx-paper.oss-cn-beijing.aliyuncs.com/IQ%E5%A4%A7%E7%A5%9E%E5%8D%9A%E5%AE%A2%E9%98%85%E8%AF%BB/%E5%9B%BE%E7%89%87/IQ%E5%88%86%E6%9E%90PDF%E7%BF%BB%E8%AF%91/22.PNG" style="zoom:50%;" />

它的轮廓是无比复杂的，从曼德尔布洛特在电脑显示器上观察到它的那一天起，它的轮廓就一直在研究中。下面是一个例子：

<img src="https://jmx-paper.oss-cn-beijing.aliyuncs.com/IQ%E5%A4%A7%E7%A5%9E%E5%8D%9A%E5%AE%A2%E9%98%85%E8%AF%BB/%E5%9B%BE%E7%89%87/IQ%E5%88%86%E6%9E%90PDF%E7%BF%BB%E8%AF%91/2-3.PNG" style="zoom:50%;" />

其分形轮廓充满了螺旋形、人字形、海马形和环形[5]。然而，整体实际上是由无数个圆盘组成的，这些圆盘组合在一起，形成了一个越来越小的瀑布。

<img src="https://jmx-paper.oss-cn-beijing.aliyuncs.com/IQ%E5%A4%A7%E7%A5%9E%E5%8D%9A%E5%AE%A2%E9%98%85%E8%AF%BB/%E5%9B%BE%E7%89%87/IQ%E5%88%86%E6%9E%90PDF%E7%BF%BB%E8%AF%91/2-4.PNG" style="zoom:50%;" />

### 2.3 观察结果

​		由于曼德尔布罗特集是由动态系统的C平面上的点组成的，而动态系统的如下：
$$
\begin{align}
&Z_{n+1}=Z_n^2+C\\
&Z_0=0
\end{align}
$$
​		它以一种非分歧的方式表现，人们想知道“非分歧”是由什么组成的。 与我们在大多数大学教科书中所相信的相反，“没有分歧”并不等同于“收敛”。 可能的行为比这两个简单的极端情况要丰富。混沌理论是分形几何学中最年轻的数学潮流之一，它对这种行为有一个答案，一个例子就是曼德尔布罗特集：两种理论都认为它具有丰富的此类行为，这种行为是周期性混乱的运动。 后者无疑是最令人惊讶的，它允许动态系统经历无穷无尽的状态，从一个状态跳转到另一个状态，而没有任何顺序，也不会发散。

​		因此，根据定义，曼德尔布罗特集是由产生收敛、周期性或混沌序列的点组成。不过，总的来说，在一般情况下，我们不需要太过深入的讨论，就可以将这三种行为 "合二为一"，认为它们都是周期性的。既然如此，收敛行为将是周期为1的周期性行为，而混沌行为则可能是一种无限期的行为。在这种行为中，一个人永远不会经历两次相同的状态。 

​		图2.5显示了前一次迭代产生的两个C值产生的序列中的点的行为，这两个C值产生的序列成为周期性的序列。在左边的图中，观察到周期等于3的周期行为，而在右边的图中，该序列收敛到周期等于5的周期。

<img src="https://jmx-paper.oss-cn-beijing.aliyuncs.com/IQ%E5%A4%A7%E7%A5%9E%E5%8D%9A%E5%AE%A2%E9%98%85%E8%AF%BB/%E5%9B%BE%E7%89%87/IQ%E5%88%86%E6%9E%90PDF%E7%BF%BB%E8%AF%91/2-5.PNG" style="zoom:50%;" />

​		因此，对曼德尔布罗的整体的第一项研究包括确定他的哪些点产生了一定周期的周期序列。从图2.2和2.4中，以及从计算机上的曼德尔布罗特集的研究中，我们可以肯定，这是由无限多的圆形 "灯泡 "组成的，这些灯泡通过脆弱的切点相互粘连，维持着集的内聚力。另一方面，由无限细的球茎线连接在一起，不时出现完整的曼德尔布罗特集的 "副本 "与各自的簇的 "副本"。

​		事实证明，正如现在将要描述的那样，同一时期的周期性行为的点被挤进这些球体中，这样一来，某一个球体内部的所有点都会产生一定时期的周期性行为，这可以理解为球体的"特征"。

​		这一事实很容易通过计算机程序来验证，该程序可以计算出曼德尔布罗特集的每一个点，从动态系统中产生的数值序列的周期，为该点产生的动态系统所产生的周期。图2.6显示了这样一个实验的结果，如图例所示，同一时期的周期性序列的起始点被标记为不同的颜色。

<img src="https://jmx-paper.oss-cn-beijing.aliyuncs.com/IQ%E5%A4%A7%E7%A5%9E%E5%8D%9A%E5%AE%A2%E9%98%85%E8%AF%BB/%E5%9B%BE%E7%89%87/IQ%E5%88%86%E6%9E%90PDF%E7%BF%BB%E8%AF%91/2-6.PNG" style="zoom:67%;" />

​		首先引起人们注意的是，这个数字似乎很明显，可能有几个同时期的圆。例如，在N=3的情况下，可以看到两个圆分别放在大灰色圆的上方和下方。如果我们再仔细观察一下，就会发现最左边还有一个N=3的圆。同样的，我们可以看到，我们可以看到3个N=4和至少6个N=5的圆形，但似乎只有一个N=1的圆和N=2的圆。

​		准确计算分布在曼德尔布罗特复合体无限分形结构中的现有圆的数量，是本工作的一个章节的一部分，并将引导我们推导出一个准确的公式来计算这个数量，作为N的函数。





## 3. 相对实轴对称的曼德布罗集

​		这个简单的论证结果将对我们在以后的论证中有所帮助，并从数学上印证了这样一个实验事实，即曼德尔布罗特集似乎是相对于实轴对称的。根据定义，曼德尔布罗特集是C平面上的迭代的点的集合：
$$
\begin{align}
&Z_{n+1}=Z_n^2+C\\
&Z_0=0
\end{align}
$$
迭代的行为完全由C决定，Zn值只依赖于C，实际上，Zn的依赖性可以用C的多项式来表示，其度数随着迭代次数（n）的增加而增加。 
$$
\begin{align}
Z_0&=0\\
Z_1&=C\\
Z_2&=C^2+C\\
Z_3&=C^4+2C^3+C^2+C\\
\cdots
\end{align}
$$
对于C的某些值，随着迭代次数(n)的增加，多项式(它们的模数)将无限制不断增加，而其他的多项式将停滞不前。根据集的定义，这些点被说成是属于曼德尔布罗特集的那些点。

 一般来说，Zn和C会有一个实数和一个虚数部分。后面是证明这个结论，不是很难，就不写了。（实际上Typora出问题了，明明保存了，却实际没有保存，我不想重新翻译了，就通过吧）





## 4.   BULB定理 

### 4.1 目标

​		本节的初步目的是为了得到一种分析方法来确定C平面的哪些点属于曼德布罗集，这相当于确定它的球茎的位置。由于这个集合的极限是混沌的，只能通过试验和试错来确定，所以我们在这里开始的探索，只是为了对这个集合做出一个方法。这个搜索会给我们提供有价值的信息。最终的目标是确定属于曼德布罗集合的C平面区域，换句话说，就是在迭代下的C平面区域 
$$
\begin{align}
Z_{n+1}&=Z_n^2+C\\\
Z_0&=0
\end{align}
$$
==最终汇聚成周期性或混沌轨道。如前所述，我们可以把普通收敛看作是周期等于1的周期性收敛的特殊情况，把混沌的情况看作是周期无限的周期性收敛的极端情况。==

在这种情况下，我们必须研究这种迭代产生的值的序列，我们也将表示为 
$$
Z_{n+1}=f_c(Z_n),Z_n=0
$$
其中
$$
f_c(Z)=Z^2+C
$$
并生成一个值序列Zn = { Z0, Z1, Z2, Z3, .... Zn }。为了使迭代抛出的值序列收敛到周期性轨道，必须有一个“N"，即
$$
Z_i=Z_{i+N}
$$
一旦满足这个条件，也就是说，一旦序列中的一个值（Z~i+N~）重复了序列本身的前一个值（Z~i~），那么在其余的迭代中，轨道将是周期性的。一个假设性的例子，让我们更容易想象出我们所描述的情况。

![](https://jmx-paper.oss-cn-beijing.aliyuncs.com/IQ%E5%A4%A7%E7%A5%9E%E5%8D%9A%E5%AE%A2%E9%98%85%E8%AF%BB/%E5%9B%BE%E7%89%87/IQ%E5%88%86%E6%9E%90PDF%E7%BF%BB%E8%AF%91/3-3.PNG)

在图像上表示如下：

<img src="https://jmx-paper.oss-cn-beijing.aliyuncs.com/IQ%E5%A4%A7%E7%A5%9E%E5%8D%9A%E5%AE%A2%E9%98%85%E8%AF%BB/%E5%9B%BE%E7%89%87/IQ%E5%88%86%E6%9E%90PDF%E7%BF%BB%E8%AF%91/3-4.PNG" style="zoom:67%;" />

### 4.2 流程

​		对于我们的搜索，我们利用了一个事实，即我们知道x = g（x）形式的一系列迭代序列必须满足的条件，以便其收敛到某个值（称为迭代的固定点）。 当g（x）在``x的可能值范围内连续时，此条件由``"不动点定理''给出：
$$
|g^`(x)|\leq 1,\forall x
$$
满足不等式的点“ x”在应用程序“ g”的收敛区域内，其极限是满足相等性的点。为了能够用这个定理来寻找-fc下的C点引起周期性轨道，我们使用下面的观察结果。让我们假设在-fc变换下的N周期轨道。
$$
\{\cdots,Z_i,f_c^2(Z_i),\cdots,f_c^{N-1}(Z_i),f_c^N(Z_i)=Z_i,\cdots\}
$$
那么，每次迭代应用"-fc" N次形成的序列就形成了子值恒定的序列。也就是说，在N个点中的任何一个点上应用"-fc "所产生的序列构成了常数序列。图4.2显示了这一观察，对于N=3的情况。

<img src="https://jmx-paper.oss-cn-beijing.aliyuncs.com/IQ%E5%A4%A7%E7%A5%9E%E5%8D%9A%E5%AE%A2%E9%98%85%E8%AF%BB/%E5%9B%BE%E7%89%87/IQ%E5%88%86%E6%9E%90PDF%E7%BF%BB%E8%AF%91/3-5.PNG" style="zoom:67%;" />

因此，如果通过定点定理，我们找到由"-fc "产生的这些序列中的某些序列收敛的区域，我们将得到"-fc "的迭代产生周期N的周期性轨道的区域，这正是我们要寻找的区域。

### 4.3 定理（一）的证明 

如前所述，我们设置一个N，形成应用fc 并寻求它们的收敛状态。

