# 2. Policy Gradient

## 2.1 Origin Policy Gradient

![](http://oss.hackslog.cn/imgs/075626.jpg)

在alpha go场景中，actor决定下哪个位置，env就是你的对手，reward是围棋的规则。强化学习三大基本组件里面，env和reward是事先给定的，我们唯一能做的就是通过调整actor，使得到的累积reward最大化。



![](http://oss.hackslog.cn/imgs/075657.jpg)



一般把actor的策略定义成Policy，数学符号为$\pi$，参数是$\theta$，本质是一个NN（神经网络）。



那么针对Atari游戏：输入游戏的画面，Policy $\pi$输出各个动作的概率，agent根据这个概率分布采取行动。通过调整$\theta$, 我们就可以调整策略的输出。



![](http://oss.hackslog.cn/imgs/075712.jpg)_page-0007)



每次采取一个行动会有一个reward



![](http://oss.hackslog.cn/imgs/075809.jpg)

玩一场游戏叫做一个episode，actor存在的目的就是最大化所能得到的return，这个return指的是每一个时间步得到的reward之和。注意我们期望最大化的是return，不是一个时刻的reward。



如果max的目标是当下时刻的reward，那么在Atari游戏中如果agent在某个s下执行开火，得到了较大的reward，那么可能agent就会一直选择开火。并不代表，最终能够取得游戏的胜利。



那么，怎么得到这个actor呢？



![](http://oss.hackslog.cn/imgs/075903.jpg)



先定义玩一次游戏，即一个episode的游戏记录为trajectory $\tau$，内容如图所示，是s-a组成的序列对。



假设actor的参数$\theta$已经给定，则可以得到每个$\tau$出现的概率。这个概率取决于两部分，$p\left(s_{t+1} | s_{t}, a_{t}\right)$部分由env的机制决定，actor没法控制，我们能控制的是$p_{\theta}\left(a_{t} | s_{t}\right)$ 由$\pi$的参数$\theta$决定。



![](http://oss.hackslog.cn/imgs/075918.jpg)



定义$R(\tau)$ 为一个episode的总的reward，即每个时间步下的即时reward相加，我习惯表述为return。



定义$\bar{R}_{\theta}$ 为$R(\tau)$的期望，等价于将每一个轨迹$\tau$出现的概率乘与其return，再求和。



由于$R(\tau)$是一个随机变量，因为actor本身在给定同样的state下会采取什么行为具有随机性，env在给定行为下输出什么state，也是随机的，所以只能算$R(\tau)$的期望。



我们的目标就变成了最大化Expected Reward，那么如何最大化？



![](http://oss.hackslog.cn/imgs/075954.jpg)



优化算法是梯度更新，首先我们先计算出$\bar{R}_{\theta}$ 对$\theta$的梯度。



从公式中可以看出$R(\tau)$可以是不可微的，因为与参数无关，不需要求导。



第一个改写（红色部分）：将加权求和写成期望的形式。



第二个近似：实际上没有办法把所有可能的轨迹（游戏记录）都求出来，所以一般是采样N个轨迹



第三个改写：将$p_{\theta}\left(\tau^{n}\right)$的表达式展开(前2页slide)，去掉跟$\theta$无关的项（不需要求导），则可达到最终的简化结果。具体如下：首先用actor采集一个游戏记录



![image-20191029215615001](http://oss.hackslog.cn/imgs/2019-11-06-080254.png)



![image-20191029220147651](http://oss.hackslog.cn/imgs/2019-11-06-080301.png)



最终得到的公式相当的直觉，在s下采取了a导致最终结果赢了，那么return就是正的，也就是会增加相应的s-a出现的概率P。



上面的公式推导中可能会有疑问，为什么要引入log？再乘一个概率除一个概率？原因非常的直觉，如下：如果动作b本来出现的次数就多，那么在加权平均所有的episode后，参数会偏好执行动作b，而实际上动作b得到的return比a低，所以除掉自身出现的概率，以降低其对训练的影响。



![image-20191029220546039](http://oss.hackslog.cn/imgs/2019-11-06-080313.png)



那么，到底是怎么更新参数的呢？



![](http://oss.hackslog.cn/imgs/080148.jpg)

首先会拿agent跟环境互动，收集大量游戏记录，然后把每一个游戏记录拿到右边，计算一个参数theta的更新值，更新参数后，再拿新的actor去收集游戏记录，不断循环。



注意：一般采样的数据只会用一次，用完就丢弃



![](http://oss.hackslog.cn/imgs/2019-11-06-080339.jpg)



具体实现：可当成一个分类任务，只是分类的结果不是识别object，是给出actor要执行的动作。



如何构建训练集？ 采样得到的a，作为ground truth。然后去最小化loss function。



一般的分类问题loss function是交叉熵，在强化学习里面，只需要在前面乘一个weight，即交叉熵乘一个return。



实现的过程中还有一些tips可以提高效果：



![](http://oss.hackslog.cn/imgs/2019-11-06-080618.jpg)



如果 reward都是正的，那么理想的情况下：reward从大到小 b>a>c, 出现次数 b>a>c, 经过训练以后，reward值高的a，c会提高出现的概率，b会降低。但如果a没有采样到，则a出现的概率最终可能会下降，尽管a的reward高。



解决方法：增加一个baseline，用r-b作为新的reward，让其有正有负。最简单的做法是b取所有轨迹的平均回报。

一般r-b叫做优势函数Advantage Functions。我们不需要描述一个行动的绝对好坏，而只需要知道它相对于平均水平的优势。



![](http://oss.hackslog.cn/imgs/2019-11-06-080716.jpg)



在这个公式里面，对于一个轨迹，每一个s-a的pair都会乘同一个weight，显然不公平，因为一局游戏里面往往有好的pair，有对结果不好的pair。所以我们希望给每一个pair乘不同的weight。整场游戏结果是好的，不代表每一个pair都是好的。如果sample次数够多，则不存在这个问题。



解决思路：在执行当下action之前的事情跟其没有关系，无论得到多少功劳都跟它没有关系，只考虑在当下执行pair之后的reward，这才是它真正的贡献。把原来的总的return，换成未来的return。



如图：对于第一组数据，在($s_a$,$a_1$)时候总的return是+3，那么如果对每一个pair都乘3，则($s_b$,$a_2$)会认为是有效的，但如果使用改进的思路，将其乘之后的return，即-2，则能有效抑制该pair对结果的贡献。



再改进：加一个折扣系数，如果时间拖得越长，对于越之后的reward，当下action的功劳越小。



![](http://oss.hackslog.cn/imgs/2019-11-06-080806.jpg)



我们将R-b 记为 A，意义就是评价当前s执行动作a，相比于采取其他的a，它有多好。之后我们会用一个critic网络来估计这个评价值。

## 2.2 PPO

PPO算法是PG算法的变形,目的是把在线的学习变成离线的学习。

核心的idea是对每一条经验（又称轨迹，即一个episode的游戏记录）不止使用一次。

简单理解：在线学习就是一边玩一边学，离线学习就是先看着别人玩进行学习，之后再自己玩

![](http://oss.hackslog.cn/imgs/2019-11-06-080842.jpg)



![](http://oss.hackslog.cn/imgs/2019-11-06-080853.jpg)



Motivation：每次用$\pi_\theta$去采样数据之后，$\pi_\theta$都会更新，接下来又要采样新的数据。以至于PG算法大部分时间都在采样数据。那么能不能将这些数据保存下来，由另一个$\pi_{\theta'}$去更新参数？那么策略$\pi_\theta$采样的数据就能被$\pi_{\theta'}$多次利用。引入统计学中的经典方法：



重要性采样：如果想求一个函数的期望，但无法积分，则可以通过采样求平均的方法来近似，但是如果p分布不知道（无法采样），我们知道q分布，则如上图通过一个重要性权重，用q分布来替代p分布进行采样。这个重要性权重的作用就是修正两个分布的差异。



![](http://oss.hackslog.cn/imgs/2019-11-06-080922.jpg)



存在的问题：如果p跟q的差异比较大，则方差会很大



![](http://oss.hackslog.cn/imgs/2019-11-06-080951.jpg)



如果sample的次数不够多，比如按原分布p进行采样，最终f的期望值是负数（大部分概率都在左侧，左侧f是负值），如果按q分布进行sample，只sample到右边，则f就一直是正的，严重偏离原来的分布。当然采样次数够多的时候，q也sample到了左边，则p/q这个负weight非常大，会平衡掉右边的正值，会导致最终计算出的期望值仍然是负值。但实际上采样的次数总是有限的，出现这种问题的概率也很大。



先忽略这个问题，加入重要性采样之后，训练变成了离线的



![](http://oss.hackslog.cn/imgs/2019-11-06-081110.jpg)



离线训练的实现：用另一个policy2与环境做互动，采集数据，然后在这个数据上训练policy1。尽管2个采集的数据分布不一样，但加入一个重要性的weights，可以修正其差异。等policy1训练的差不多以后，policy2再去采集数据，不断循环。



![](http://oss.hackslog.cn/imgs/2019-11-06-081122.jpg)



由于我们得到的$A^{\theta}\left(s_{t}, a_{t}\right)$（执行当下action后得到reward-baseline）是由policy2采集的数据观察得到的，所以 $A^{\theta}\left(s_{t}, a_{t}\right)$的参数得修正为$\theta'$



根据$\nabla f(x)=f(x) \nabla \log f(x)$反推目标函数$J$，注意要优化的参数是$\theta$ ，$\theta’$只负责采集数据。



利用 $\theta’$采集的数据来训练$\theta$，会不会有问题？（虽然有修正，但毕竟还是不同） 答案是我们需要保证他们的差异尽可能的小，那么在刚刚的公式里再加入一些限制保证其差异足够小，则诞生了 PPO算法。



![](http://oss.hackslog.cn/imgs/2019-11-06-081210.jpg)



引入函数KL，KL衡量两个分布的距离。注意：不是参数上的距离，是2个$\pi$给同样的state之后基于各自参数输出的action space的距离



加入KL的公式直觉的理解：如果我们学习出来的$\theta$跟$\theta'$越像，则KL越小，J越大。我们的学习目标还是跟原先的PG算法一样，用梯度上升训练，最大化J。这个操作有点像正则化，用来解决重要性采样存在的问题。



TRPO是PPO是前身，把KL这个限制条件放在优化的目标函数外面。对于梯度上升的优化过程，这种限制比较难处理，使优化变得复杂，一般不用。



![](http://oss.hackslog.cn/imgs/2019-11-06-081228.jpg)



实现过程：初始化policy参数，在每一个迭代里面，用$theta^k$采集很多数据，同时计算出奖励A值，接着train这个数据，更新$\theta$优化J。由于是离线训练，可以多次更新后，再去采集新的数据。



有一个trick是KL的权重beta也可以调整，使其更加的adaptive。



![](http://oss.hackslog.cn/imgs/2019-11-06-081245.jpg)



![](http://oss.hackslog.cn/imgs/2019-11-06-081315.jpg)



实际上KL也是比较难计算的，所以有了PPO2算法，不计算KL，通过clip达到同样效果。



clip(a, b, c): if a<b => b  If a>c => c If b<a<c => a



看图：绿色：min里面的第一项，蓝色：min里面的第二项，红色 min的输出



这个公式的直觉理解：希望$\theta$与$\theta^k$在优化之后不要差距太大。如果A>0，即这个state-action是好的，所以需要增加这个pair出现的几率，所以在max J的过程中会增大$\frac{p_{\theta}\left(a_{t} | s_{t}\right)}{p_{\theta^{k}}\left(a_{t} | s_{t}\right)}$, 但最大不要超过1+eplison，如果A<0，不断减小，小到1-eplison，始终不会相差太大。



![](http://oss.hackslog.cn/imgs/2019-11-06-081323.jpg)

PG算法效果非常不稳定，自从有了PPO，PG的算法可以在很多任务上work。

