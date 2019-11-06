# 6. Imitation Learning 

模仿学习，又叫学徒学习，反向强化学习

之前介绍的强化学习都有一个reward function，但生活中大多数任务无法定义reward，或者难以定义。但是这些任务中如果收集很厉害的范例（专家经验）比较简单，则可以用模仿学习解决。

![](http://oss.hackslog.cn/imgs/2019-11-06-092747.jpg)

## 6.1 Behavior Cloning

本质是有监督学习

![0004](http://oss.hackslog.cn/imgs/2019-11-06-092759.jpg)

![0005](http://oss.hackslog.cn/imgs/2019-11-06-092817.jpg)

存在问题：training data里面没有撞墙的case，则agent遇到这种情况不知如何决策

![0006](http://oss.hackslog.cn/imgs/2019-11-06-092821.jpg)

一个直觉的解决方法是数据增强：每次通过牺牲一个专家，学会了一种新的case，策略$\pi$得到了增强。

![](http://oss.hackslog.cn/imgs/2019-11-06-092831.jpg)

行为克隆还存在一个关键问题：agent不知道哪些行为对结局重要，哪些不重要。由于是采样学习，有可能只记住了多余的无用的行为。

![0009](http://oss.hackslog.cn/imgs/2019-11-06-092847.jpg)

同时也由于RL的训练数据不是独立同分布，当下的action会影响之后的state，所以不能直接套用监督学习的框架。

为了解决这些问题，就有了反向强化学习，现在一般说模仿学习指的就是反向强化学习。

## 6.2 Inverse RL

![0011](http://oss.hackslog.cn/imgs/2019-11-06-092859.jpg)

之前的强化学习是reard和env通过RL 学到一个最优的actor。

反向强化学习是，假设有一批expert的数据，通过env和IRL推导expert因为什么样子的reward function才会采取这样的行为。

好处：也许expert的行为复杂但reward function很简单。拿到这个reward function后我们就可以训练出好的agent。

![0012](http://oss.hackslog.cn/imgs/2019-11-06-092907.jpg)

IRL的框架：先射箭 再画靶。

具体过程：

Expert先跟环境互动，玩N场游戏，存储记录，我们的actor $ \pi$也去互动，生成N场游戏记录。接下来定义一个reward function $R$，保证expert的$R$比我们的actor的$R$大就行。再根据定义的的$R$用RL的方法去学习一个新的actor ，这个过程也会采集新的游戏记录，等训练好这个actor，也就是当这个actor可以基于$R$获得高分的时候，重新定义一个新的reward function$R'$，让expert的$R'$大于agent，不断循环。

![0013](http://oss.hackslog.cn/imgs/2019-11-06-092917.jpg)

IRL与GAN的框架是一样的，学习 一个 reward function相当于学习一个判别器，这个判别器给expert高分，给我们的actor低分。

一个有趣的事实是给不同的expert，我们的agent最终也会学会不同的策略风格。如下蓝色是expert的行为，红色是学习到的actor的行为。

![](http://oss.hackslog.cn/imgs/2019-11-06-092932.jpg)

针对训练robot的任务：

IRL有个好处是不需要定义规则让robot执行动作，人给robot示范一下动作即可。但robot学习时候的视野跟它执行该动作时候的视野不一致，怎么把它在第三人称视野学到的策略泛化到第一人称视野呢？

![](http://oss.hackslog.cn/imgs/2019-11-06-092943.jpg)

![0019](http://oss.hackslog.cn/imgs/2019-11-06-092953.jpg)

解决思路跟好奇心机制类似，抽出视野中不重要的因素，让第一人称和第三人称视野中的state都是有用的，与action强相关的。