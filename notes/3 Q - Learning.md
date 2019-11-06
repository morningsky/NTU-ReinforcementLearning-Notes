# 3. Q - Learning

## 3.1 Q-learning

在之前的policy-based算法里，我们的目标是learn 一个actor，value-based的强化学习算法目标是learn一个critic。 

定义一个Critic，也就是状态值函数$V^{\pi}(s)$，它的值是：当使用策略$\pi$进行游戏时，在观察到一个state s之后，环境输出的累积的reward值的期望。注意取决于两个值，一个是state s，一个是actor$\pi$。

![](http://oss.hackslog.cn/imgs/2019-11-06-081510.jpg)

![0004](http://oss.hackslog.cn/imgs/2019-11-06-081524.jpg)

如果是不同的actor，在同样的state下，critic给出的值也是不同的。那么怎么估计出这个函数V呢？

主要有MC和TD的方法，实际上还有DP的方法，但是用DP求解需要整个环境都是已知的。而在强化学习的大部分任务里，都是model-free的，需要agent自己去探索环境。

![0005](http://oss.hackslog.cn/imgs/2019-11-06-081539.jpg)

MC：直接让agent与环境互动，统计计算出在$S_a$之后直到一个episode结束的累积reward作为$G_a$。

训练的目标就是让$V^{\pi}(s)$的输出尽可能的接近$G_a$。

![0006](http://oss.hackslog.cn/imgs/2019-11-06-081554.jpg)

MC每次必须把游戏玩到结束，TD不需要把游戏玩到底，只需要玩了一次游戏，有一个状态的变化。

那么训练的目标就是让$V^{\pi}(s_t)$  和$V^{\pi}(s_t+1)$的差接近$r_t$

![0007](http://oss.hackslog.cn/imgs/2019-11-06-081607.jpg)

MC方差大，因为$r$是一个随机变量，MC方法中的$G$是$r$之和，而TD方法只有$r$是随机变量，r的方差比G小。但TD方法的$V^{\pi}$有可能估计的不准。

![0008](http://oss.hackslog.cn/imgs/2019-11-06-081621.jpg)

用MC和TD估计的结果不一样

![0009](http://oss.hackslog.cn/imgs/2019-11-06-081633.jpg)

定义另一种Critic，状态-动作值函数$Q^{\pi}(s,a)$，有的地方叫做Q-function，输入是一个pair $(s,a)$，意思是用$\pi$玩游戏时，在s状态下强制执行动作a（策略$\pi$在s下不一定会执行a），所得到的累积reward。

有两种写法，输入pair，输出Q，此时的Q是一个标量。

另一种是输入s，输出所有可能的action的Q值，此时Q是一个向量。

那么Critic到底怎么用呢？

![](http://oss.hackslog.cn/imgs/2019-11-06-081645.jpg)

Q-learning的过程：

初始化一个actor $\pi$去收集数据，然后learn一个基于$ \pi$的Q-function，接着寻找一个新的比原来的$\pi$要好actor , 找到后更新$\pi$，再去寻找新的Q-function，不断循环，得到更好的policy。

可见Q-learning的核心思想是先找到最优的Q-function，再通过这个Q-function得出最优策略。而Policy-based的算法是直接去学习策略。这是本质区别。

那么，怎么样才算比原来的好？

![0012](http://oss.hackslog.cn/imgs/2019-11-06-081701.jpg)

定义好的策略：对所有可能的s而言，$V_\pi(s)$一定小于$V_\pi'(s)$，则$V_\pi'(s)$就是更好的策略。

$\pi'(s)$的本质：假设已经学习到了一个actor $\pi$的Q-function，给一个state，把所有可能的action都代入Q，执行那个可以让Q最大的action。

注意：实际上，给定一个s，$ \pi$不一定会执行a，现在的做法是强制执行a，计算执行之后玩下去得到的reward进行比较。

在实现的时候$\pi'$没有额外的参数，依赖于Q。并且当动作是连续值的时候，无法进行argmax。

那么， 为什么actor $\pi’$能被找到？

![0013](http://oss.hackslog.cn/imgs/2019-11-06-081720.jpg)

上面是为了证明：只要你估计出了一个actor的Q-function，则一定可以找到一个更好的actor。

核心思想：在一个episode中某一步把$\pi$换成了$ \pi'$比完全follow $ \pi$，得到的奖励期望值会更大。

注意$r_{t+1}$指的是在执行当下$a_t$得到的奖励，有的文献也会写成$r_t$

训练的时候有一些Tips可以提高效率：

![0014](http://oss.hackslog.cn/imgs/2019-11-06-081740.jpg)

Tips 1 引入target网络

训练的时候，每次需要两个Q function（两个的输入不同）一起更新，不稳定。 一般会固定一个Q作为Target，产生回归任务的label，在训练N次之后，再更新target的参数。回归任务的目标，让$Q^\pi(s_t,a_t)$与$\mathrm{Q}^{\pi}\left(s_{t+1}, \pi\left(s_{t+1}\right)\right))+r$越来越接近，即降低mse。最终希望训练得到的$Q^\pi$能直接估计出这个$(s_t,a_t)$未来的一个累积奖励。

注意：target网络的参数不需要训练，直接每隔N次复制Q的参数。训练的目标只有一个 Q。

![0015](http://oss.hackslog.cn/imgs/2019-11-06-081754.jpg)

Tips2 改进探索机制

PG算法，每次都会sample新的action，随机性比较大，大概率会尽可能的覆盖所有的动作。而之前的Q-learning，策略的本质是绝对贪婪策略，那么如果有的action没有被sample到，则可能之后再也不会选择这样的action。这种探索的机制（收集数据的方法）不好，所以改进贪心算法，让actor每次会$\varepsilon$的概率执行随机动作。

![0016](http://oss.hackslog.cn/imgs/2019-11-06-081820.jpg)

![0017](http://oss.hackslog.cn/imgs/2019-11-06-081833.jpg)

Tips 3 引入记忆池机制

将采集到的一些数据收集起来，放入replay buffer。好处：

1.可重复使用过去的policy采集的数据，降低agent与环境互动的次数，加快训练效率 。

2.replay buffer里面包含了不同actor采集的数据，这样每次随机抽取一个batch进行训练的时候，每个batch内的数据会有较大的差异（数据更加diverse），有助于训练。

那么，当我们训练的目标是$ \pi$的Q-function，训练数据混杂了$\pi’$,$\pi’'$,$\pi’''$采集的数据 有没有问题呢？没有，不是因为这些$ \pi$很像，主要原因是我们采样的不是一个轨迹，只是采样了一笔experience($s_t,a_t,r_t,s_{t+1}$)。这个理论上证明是没有问题的，很难解释...

![0018](http://oss.hackslog.cn/imgs/2019-11-06-081844.jpg)

采用了3个Tips的Q-learning训练过程如图：

注意图中省略了一个循环，即存储了很多笔experience之后才会进行sample。相比于原始的Q-learning，每次sample是从replay buff里面随机抽一个batch，然后计算用绝对贪心策略得到Q-target的值作为label，接着在回归任务中更新Q的参数。每训练多步后，更新Q-target的参数。

## 3.2 Tips of Q-learning

![](http://oss.hackslog.cn/imgs/2019-11-06-081854.jpg)

DQN估计出的值一般都高于实际的值，double DQN估计出的值与实际值比较接近。

![0021](http://oss.hackslog.cn/imgs/2019-11-06-081906.jpg)

Q是一个估计值，被高估的越多，越容易被选择。

![0022](http://oss.hackslog.cn/imgs/2019-11-06-081948.jpg)

Double的思想有点像行政跟立法分权。

用要训练的Q-network去选择动作，用固定不动的target-network去做估计，相比于DQN,只需要改一行代码！

![0023](http://oss.hackslog.cn/imgs/2019-11-06-081958.jpg)

改了network架构，其他没动。每个网络结构的输出是一个标量+一个向量

![0024](http://oss.hackslog.cn/imgs/2019-11-06-082009.jpg)

比如下一时刻，我们需要把3->4, 0->-1,那么Dueling结构里会倾向于不修改A，只调整V来达到目的，这样只需要把V中 0->1, 如果Q中的第三行-2没有被sample到，也进行了更新，提高效率，减少训练次数。

![0025](http://oss.hackslog.cn/imgs/2019-11-06-082019.jpg)

实际实现的时候，通过添加了限制条件，也就是把A normalize，使得其和为0，这样只会更新V。

这种结构让DQN也能处理连续的动作空间。

![0028](http://oss.hackslog.cn/imgs/2019-11-06-082253.jpg)

加入权重的replay buffer

motivation：TD error大的数据应该更可能被采样到

注意论文原文实现的细节里，也修改了参数更新的方法

![0029](http://oss.hackslog.cn/imgs/2019-11-06-082300.jpg)

原来收集一条experience是执行一个step，现在变成执行N个step。相比TD的好处：之前只sample一个$(s_t,a_t)$pair，现在sample多个才估测Q值，估计的误差会更小。坏处，与MC一样，reward的项数比较多，相加的方差更大。 调N就是一个trade-off的过程。

![0030](http://oss.hackslog.cn/imgs/2019-11-06-082309.jpg)

在Q-function的参数空间上+noise

比较有意思的是，OpenAI DeepMind几乎在同一个时间发布了Noisy Net思想的论文。

![0031](http://oss.hackslog.cn/imgs/2019-11-06-082316.jpg)

在同一个episode里面，在动作空间上加噪声，会导致相同state下执行的action不一样。而在参数空间加噪声，则在相同或者相似的state下，会采取同一个action。 注意加噪声只是为了在不同的episode的里面，train Q的时候不会针对特定的一个state永远只执行一个特定的action。

![0033](http://oss.hackslog.cn/imgs/2019-11-06-082325.jpg)

带分布的Q-function

Motivation：原来计算Q-function的值是通过累积reward的期望，也就是均值，但实际上累积的reward可能在不同的分布下会得到相同的Q值。

注意：每个Q-function的本质都是一个概率分布。

![0034](http://oss.hackslog.cn/imgs/2019-11-06-082332.jpg)

让$Q^ \pi$直接输出每一个Q-function的分布，但实际上选择action的时候还是会根据mean值大的选。不过拥有了这个分布，可以计算方差，这样如果有的任务需要在追求回报最大的同时降低风险，则可以利用这个分布。

![0036](http://oss.hackslog.cn/imgs/2019-11-06-082339.jpg)

![0037](http://oss.hackslog.cn/imgs/2019-11-06-082345.jpg)

Rainbow：集成了7种升级技术的DQN

上图是一个一个改进拿掉之后的效果，看紫色似乎double 没啥用，实际上是因为有Q-function的分布存在，一般不会过高估计Q值，所以double 意义不大。

直觉的理解：使用分布DQN，即时Q值被高估很多，由于最终只会映射到对应的分布区间，所以最终的输出值也不会过大。

## 3.3 Q-learning in continuous actions

在出现PPO之前, PG的算法非常不稳定。DQN 比较稳定，也容易train，因为DQN是只要估计出Q-function，就能得到好的policy，而估计Q-function就是一个回归问题，回归问题比较容易判断learn的效果，看mse。问题是Q-learning不太容易处理连续动作空间。比如汽车的速度，是一个连续变量。

![0039](http://oss.hackslog.cn/imgs/2019-11-06-082354.jpg)

当动作值是连续时，怎么解argmax：

1. 通过映射，强行离散化

2. 使用梯度上升解这个公式，这相当于每次train完Q后，在选择action的时候又要train一次网络，比较耗时间。

![0040](http://oss.hackslog.cn/imgs/2019-11-06-082404.jpg)

3. 设计特定的网络，使得输出还是一个标量。

![0042](http://oss.hackslog.cn/imgs/2019-11-06-082433.jpg)

最有效的解决方法是，针对连续动作空间，不要使用Q-learning。使用AC算法！