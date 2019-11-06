# 李宏毅深度强化学习 笔记

课程主页：[NTU-MLDS18](http://speech.ee.ntu.edu.tw/~tlkagk/courses_MLDS18.html)

视频：[youtube](https://www.youtube.com/playlist?list=PLJV_el3uVTsODxQFgzMzPLa16h6B8kWM_) [B站](https://www.bilibili.com/video/av24724071/?spm_id_from=333.788.videocard.4)

参考资料： [作业代码参考](https://github.com/JasonYao81000/MLDS2018SPRING/tree/master/hw4)  [纯numpy实现非Deep的RL算法](https://github.com/ddbourgin/numpy-ml/tree/master/numpy_ml/rl_models) [OpenAI tutorial](https://github.com/openai/spinningup/tree/master/docs)

# 1. Introduction

![1](http://oss.hackslog.cn/imgs/075034.png)



这门课的学习路线如上，强化学习是作为单独一个模块介绍。李宏毅老师讲这门课不是从MDP开始讲起，而是从如何获得最大化奖励出发，直接引出Policy Gradient（以及PPO），再讲Q-learning（原始Q-learning，DQN，各种DQN的升级），然后是A2C（以及A3C, DDPG），紧接着介绍了一些Reward Shaping的方法（主要是Curiosity，Curriculum Learning ，Hierarchical RL），最后介绍Imitation Learning (Inverse RL)。比较全面的展现了深度强化学习的核心内容，也比较直观。

![image-20191029211249836](http://oss.hackslog.cn/imgs/075024.png)

首先强化学习是一种解决序列决策问题的方法，他是通过与环境交互进行学习。首先会有一个Env，给agent一个state，agent根据得到的state执行一个action，这个action会改变Env，使自己跳转到下一个state，同时Env会反馈给agent一个reward，agent学习的目标就是通过采取action，使得reward的期望最大化。

![image-20191029211454593](http://oss.hackslog.cn/imgs/075050.png)



在alpha go的例子中，state（又称observation）为所看到的棋盘，action就是落子，reward通过围棋的规则给出，如果最终赢了，得1，输了，得-1。

下面从2个例子中看强化学习与有监督学习的区别。RL不需要给定标签，但需要有reward。

![image-20191029211749897](http://oss.hackslog.cn/imgs/075057.png)

实际上alphgo是从提前收集的数据上进行有监督学习，效果不错后，再去做强化学习，提高水平。

![image-20191029211848429](http://oss.hackslog.cn/imgs/075104.png)



人没有告诉机器人具体哪里说错了，机器需要根据最终的评价自己总结，一般需要对话好多次。所以通常训练对话模型会训练2个agent互相对话

![image-20191029212015623](http://oss.hackslog.cn/imgs/075108.png)

![image-20191029212144625](http://oss.hackslog.cn/imgs/075112.png)



一个难点是怎么判断对话的效果，一般会设置一些预先定义的规则。

![image-20191029212313069](http://oss.hackslog.cn/imgs/075117.png)

强化学习还有很多成功的应用，凡是序列决策问题，大多数可以用RL解决。

