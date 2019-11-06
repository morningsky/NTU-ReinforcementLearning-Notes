# 李宏毅深度强化学习 笔记

### 课程主页：[NTU-MLDS18](http://speech.ee.ntu.edu.tw/~tlkagk/courses_MLDS18.html)

### 视频：
- [youtube](https://www.youtube.com/playlist?list=PLJV_el3uVTsODxQFgzMzPLa16h6B8kWM_) 
- [B站](https://www.bilibili.com/video/av24724071/?spm_id_from=333.788.videocard.4)


![1](http://oss.hackslog.cn/imgs/075034.png)

这门课的学习路线如上，强化学习是作为单独一个模块介绍。李宏毅老师讲这门课不是从MDP开始讲起，而是从如何获得最大化奖励出发，直接引出Policy Gradient（以及PPO），再讲Q-learning（原始Q-learning，DQN，各种DQN的升级），然后是A2C（以及A3C, DDPG），紧接着介绍了一些Reward Shaping的方法（主要是Curiosity，Curriculum Learning ，Hierarchical Learning），最后介绍Imitation Learning (Inverse RL)。比较全面的展现了深度强化学习的核心内容，也比较直观。跟伯克利学派的课类似，与UCL上来就讲MDP，解各种value iteration的思路有较大区别。
文档中的notes以对slides的批注为主，方便在阅读slides时理解，code以纯tensorflow实现，主要参考[莫凡RL教学](https://morvanzhou.github.io/tutorials/machine-learning/reinforcement-learning/)，修正部分代码以保持前后一致性，已经加入便于理解的注释。
### 参考资料： 
[作业代码参考](https://github.com/JasonYao81000/MLDS2018SPRING/tree/master/hw4)  [纯numpy实现非Deep的RL算法](https://github.com/ddbourgin/numpy-ml/tree/master/numpy_ml/rl_models) [OpenAI tutorial](https://github.com/openai/spinningup/tree/master/docs) [莫凡RL教学](https://morvanzhou.github.io/tutorials/machine-learning/reinforcement-learning/)
- code中的tensorlayer实现来自于[Tensorlayer-RL](https://github.com/tensorlayer/tensorlayer/tree/master/examples/reinforcement_learning),比起原生tensorflow更加简洁
