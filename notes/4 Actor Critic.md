# 4. Actor Critic

## 4.1 Advantage Actor-Critic (A2C)

![](http://oss.hackslog.cn/imgs/2019-11-06-082605.jpg)

由于每次在执行PG算法之前，一般只能采样少量的数据，导致对于同一个$(s_t,a_t)$，得到的$G$的值方差很大，不稳定。那么能不能直接估计出期望值，来替代采样的结果？

![AC-4](http://oss.hackslog.cn/imgs/2019-11-06-082530.jpg)

回顾下Q-learning中的定义，我们发现：

![AC-5](http://oss.hackslog.cn/imgs/2019-11-06-082602.jpg)

PG算法中G的期望的定义恰好也是Q-learning算法中$Q^\pi(s,a)$的定义： 假设现在的policy是$ \pi$的情况下，在某一个s，采取某一个a以后得到的累积reward的期望值。

因此在这里将Q-learning引入到预估reward中，也即policy gradient和q-learning的结合,叫做Actor-Critic。

把原来的reward和baseline分别替换，PG算法中的减法就变成了$Q^{\pi_{\theta}}\left(s_{t}^{n}, a_{t}^{n}\right)-V^{\pi_{\theta}}\left(s_{t}^{n}\right)$。似乎我们需要训练2个网络？

![AC-6](http://oss.hackslog.cn/imgs/2019-11-06-082629.jpg)

实际上Q与V可以互相转化，我们只需要训练V。转化公式中为什么要加期望？在s下执行a得到的$ r_t$和$s_{t+1}$是随机的。

实际将Q变成V的操作中，我们会去掉期望，使得只需要训练（估计）状态值函数$V^\pi$，这样会导致一点偏差，但比同时估计两个function导致的偏差要好。（A3C原始paper通过实验验证了这一点）。

![AC-7](http://oss.hackslog.cn/imgs/2019-11-06-082641.jpg)

A2C的训练流程：收集数据，估计出状态值函数$V^\pi(s)$，套用公式更新策略$\pi$，再利用新的$\pi$与环境互动收集新的数据，不断循环。

![AC-8](http://oss.hackslog.cn/imgs/2019-11-06-082652.jpg)

训练过程中的2个Tips：

1. Actor与Critic的前几层一般会共用参数，因为输入都是state
2. 正则化：让采用不同action的概率尽量平均，希望有更大的entropy，这样能够探索更多情况。

## 4.2 Asynchronous Advantage Actor-Critic (A3C)

![AC-9](http://oss.hackslog.cn/imgs/2019-11-06-082709.jpg)

A3C算法的motivation：开分身学习~

![AC-10](http://oss.hackslog.cn/imgs/2019-11-06-082718.jpg)

训练过程：每个agent复制一份全局参数，然后各自采样数据，计算梯度，更新这份全局参数，然后将结果传回，复制一份新的参数。

注意：

1. 初始条件会尽量的保证多样性(Diverse)，让每个agent探索的情况更加不一样。

2. 所有的actor都是平行跑的，每个worker把各自的参数传回去然后复制一份新的全局参数。此时可能这份全局参数已经发生了改变，没有关系。

## 4.3 Pathwise Derivative Policy Gradient (PDPG)

在之前Actor-Critic框架里，Critic的作用是评估agent所执行的action好不好？那么Critic能不能不止给出评价，还给出指导意见呢？即告诉actor要怎样做才能更好？于是有了DPG算法：

![AC-12](http://oss.hackslog.cn/imgs/2019-11-06-082731.jpg)

![AC-13](http://oss.hackslog.cn/imgs/2019-11-06-082746.jpg)

在上面介绍A2C算法的motivation，主要是从改进PG算法引入。那么从Q-learning的角度来看，PDPG相当于learn一个actor，来解决argmax这个优化问题，以处理连续动作空间，直接根据输入的状态输出动作。

![AC-14](http://oss.hackslog.cn/imgs/2019-11-06-082759.jpg)

Actor+Critic连成一个大的网络，训练过程中也会采取TD-target的技巧，固定住Critic $\pi'$，使用梯度上升优化Actor

![AC-15](http://oss.hackslog.cn/imgs/2019-11-06-082809.jpg)

训练过程：Actor会学到策略$\pi$，使基于策略$\pi$，输入s可以获得能够最大化Q的action，天然地能够处理continuous的情况。当actor生成的$Q^\pi$效果比较好时，重新采样生成新的Q。有点像GAN中的判别器与生成器。

注意：从算法的流程可知，Actor 网络和 Critic 网络是分开训练的，但是两者的输入输出存在联系，Actor 网络输出的 action 是 Critic 网络的输入，同时 Critic 网络的输出会被用到 Actor 网路进行反向传播。

由于Critic模块是基于Q-learning算法，所以Q learning的技巧，探索机制，回忆缓冲都可以用上。

![AC-16](http://oss.hackslog.cn/imgs/2019-11-06-082820.jpg)

![AC-17](http://oss.hackslog.cn/imgs/2019-11-06-082830.jpg)

与Q-learning相比的改进：

- 不通过Q-function输出动作，直接用learn一个actor网络输出动作（Policy-based的算法的通用特性）。
- 对于连续变量，不好解argmax的优化问题，转化成了直接选择$\pi-target$ 输出的动作，再基于Q-target得出y。
  - 引入$\pi-target$，也使得actor网络不会频繁更新，会通过采样一批数据训练好后再更新，提高训练效率。

![AC-18](http://oss.hackslog.cn/imgs/2019-11-06-082845.jpg)

总结下：最基础的 Policy Gradient 是回合更新的，通过引入 Critic 后变成了单步更新，而这种结合了 policy 和 value 的方法也叫 Actor-Critic，Critic 有多种可选的方法。A3C在A2C的基础上引入了多个 agent 对网络进行异步更新。对于输出动作为连续值的情形，原始的输出动作概率分布的PG算法不能解决，同时Q-learning算法也不能处理这类问题，因此提出了 DPG 。