# 5. Sparse Reward 

大多数RL的任务中，是没法得到reward，reward=0，导致reward空间非常的sparse。

比如我们需要赢得一局游戏才能知道胜负得到reward，那么玩这句游戏的很长一段时间内，我们得不到reward。比如如果机器人要将东西放入杯子才能得到一个reward，尝试了很多动作很有可能都是0。

但是人可以在非常sprse的环境下进行学习，所以这一章节提出的很多算法与人的一些学习机制比较类似。

## 5.1 Reward Shaping

手动设计新的reward，让agent做的更好。但有些比较复杂的任务，需要domain knowledge去设计新的reward。

![](http://oss.hackslog.cn/imgs/2019-11-06-082912.jpg)

![](http://oss.hackslog.cn/imgs/2019-11-06-082928.jpg)

## 5.2 Curiosity

好奇心机制非常的直觉，也非常的强大。有个案例：[Happy Bird](https://github.com/pathak22/noreward-rl)

![](http://oss.hackslog.cn/imgs/2019-11-06-082959.jpg)

好奇心也是reward shaping的一种，引入一个新的reward ：ICM，同时优化2个reward。如何设计一个ICM模块，使agent拥有好奇心？

![](http://oss.hackslog.cn/imgs/2019-11-06-083015.jpg)

单独训练一个状态估计的模型，如果在某个state下采取某个action得到的下一个state难以预测，则鼓励agent进行尝试这个action。 不过有的state很难预测，但不重要。比如说某个游戏里面背景是树叶飘动，很难预测，接下来agent一直不动看着树叶飘动，没有意义。

![](http://oss.hackslog.cn/imgs/2019-11-06-083031.jpg)

再设计一个moudle，判断环境中state的重要性：learn一个feature ext的网络，去掉环境中与action关系不大的state。

原理：输入两个处理过的state，预测action，使逼近真实的action。这样使得处理之后的state都是跟agent要采取的action相关的。

## 5.3 Curriculum Learning

课程学习：为learning做规划，通常由易到难。

![](http://oss.hackslog.cn/imgs/2019-11-06-092656.jpg)

设计不同难度的课程，一开始直接把板子放入柱子，则agent只要把板子压下去就能获得reward，接着把板子的初始位置提高一些，agent有可能把板子抽出则无法获得reward，接着更general的情况，把板子放倒柱子外面，再让agent去学习。

生成课程的方法通常如下：从目标反推，越靠近目标的state越简单，不断生成难度更高的state。

![](http://oss.hackslog.cn/imgs/2019-11-06-092702.jpg)

![](http://oss.hackslog.cn/imgs/2019-11-06-092658.jpg)

## 5.4 Hierarchical RL

分层学习：把大的任务拆解成小任务

![](http://oss.hackslog.cn/imgs/2019-11-06-092745.jpg)

上层的agent给下层的agent提供一个愿景，如果下层的达不到目标，会获得惩罚。如果下层的agent得到的错误的目标，那么它会假设最初的目标也是错的。