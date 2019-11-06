import tensorflow as tf
import numpy as np
import gym
import time

#####################  hyper parameters  ####################

MAX_EPISODES = 200
MAX_EP_STEPS = 200
LR_A = 0.01    # learning rate for actor
LR_C = 0.02    # learning rate for critic
GAMMA = 0.9     # reward discount
TAU = 0.01      # soft replacement
MEMORY_CAPACITY = 10000
BATCH_SIZE = 32

RENDER = False
ENV_NAME = 'Pendulum-v0'

#pendulum 动作与状态都是连续空间
#动作空间：只有一维力矩 长度为1 虽然是连续值，但是有bound【-2，2】
#状态空间：一维速度，长度为3

###############################  DDPG  ####################################
#离线训练 单步更新 按batch更新 引入replay buffer机制
class DDPG(object):
    def __init__(self, a_dim, s_dim, a_bound,): #初始化2个网络图 注意无论是critic还是actor网络都有target-network机制 target-network不训练
        self.memory = np.zeros((MEMORY_CAPACITY, s_dim * 2 + a_dim +1), dtype=np.float32) #借鉴replay buff机制 s*2 : s, s_
        self.pointer = 0
        self.sess = tf.Session()

        self.a_dim, self.s_dim, self.a_bound = a_dim, s_dim, a_bound
        self.S = tf.placeholder(tf.float32, [None, s_dim], 's') #前面的None用来给batch size占位
        self.S_ = tf.placeholder(tf.float32, [None, s_dim], 's_')
        self.R = tf.placeholder(tf.float32, [None,1], 'r')

        with tf.variable_scope('Actor'):
            self.a = self._build_a(self.S, scope='eval', trainable=True) #要训练的pi网络，也负责收集数据 # input s, output a
            a_ = self._build_a(self.S, scope='target', trainable=False) #target网络不训练，只负责输出动作给critic # input s_, output a, get a_ for critic
        with tf.variable_scope('Critic'):
            q = self._build_c(self.S, self.a, scope='eval', trainable=True) #要训练的Q， 与target输出的q算mse（td-error）  注意这个a来自于memory
            q_ = self._build_c(self.S_, a_, scope='target', trainable=False) #这个网络不训练, 用于给出 Actor 更新参数时的 Gradient ascent 强度 即dq/da 注意这个a来自于actor要更新参数时候的a
    
        # networks parameters
        self.ae_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='Actor/eval')
        self.at_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='Actor/target')
        self.ce_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='Critic/eval')
        self.ct_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='Critic/target')

        #taget 网络更新 即从eval网络中复制参数
        self.soft_replace = [tf.assign(t, (1-TAU)*t + TAU *e) for t, e in zip(self.at_params+self.ct_params,self.ae_params+self.ce_params)]

        #训练critic网络（eval）
        q_target = self.R + GAMMA * q_ #贝尔曼公式（里面的q_来自于Q-target网络输入(s_，a_)的输出） 得出q的”真实值“ 与预测值求mse
        td_error = tf.losses.mean_squared_error(labels=q_target, predictions=q) #预测值q 来自于q-eval网络输入当前时刻的(s,a)的输出
        self.ctrain = tf.train.AdamOptimizer(LR_C).minimize(td_error, var_list = self.ce_params) #要train的是q-eval网络的参数 最小化mse

        #训练actor网络（eval）
        a_loss = -tf.reduce_mean(q) #maximize q
        self.atrain = tf.train.AdamOptimizer(LR_A).minimize(a_loss, var_list = self.ae_params) #
        
        self.sess.run(tf.global_variables_initializer())


    def choose_action(self, s):
        s = s[np.newaxis, :]
        return self.sess.run(self.a, feed_dict={self.S: s})[0]  # single action


    def learn(self):
        #每次学习都是先更新target网络参数
        self.sess.run(self.soft_replace)
        indices = np.random.choice(MEMORY_CAPACITY, size=BATCH_SIZE) 
        bt = self.memory[indices, : ] #从memory中取一个batch的数据来训练
        bs = bt[:, :self.s_dim] #a batch of state
        ba = bt[:, self.s_dim: self.s_dim + self.a_dim] #a batch of action
        br = bt[:, -self.s_dim - 1: -self.s_dim] #a batch of reward
        bs_ = bt[:, -self.s_dim:]

        #一次训练一个batch 这一个batch的训练过程中target网络相当于固定不动
        self.sess.run(self.atrain, {self.S: bs})
        self.sess.run(self.ctrain, {self.S: bs, self.a: ba, self.R: br, self.S_: bs_})
        

    def store_transition(self, s, a, r, s_): #离线训练算法标准操作
        transition = np.hstack((s, a, [r], s_))
        index = self.pointer % MEMORY_CAPACITY  # replace the old memory with new memory
        self.memory[index, :] = transition
        self.pointer += 1

    def _build_a(self, s, scope, trainable): #actor网络结构 直接输出动作确定a
        with tf.variable_scope(scope):
            net = tf.layers.dense(s, 30, activation=tf.nn.relu, name='l1', trainable=trainable)
            a = tf.layers.dense(net, self.a_dim, activation=tf.nn.tanh, name='a', trainable=trainable) #a经过了tanh 数值缩放到了【-1，1】
            return tf.multiply(a, self.a_bound, name='scaled_a') #输出的每个a值都乘边界[max,] 可以保证输出范围在【-max，max】 如果最小 最大值不是相反数 得用clip正则化

    def _build_c(self, s, a, scope, trainable): #critic网络结构 输出Q(s,a)
        with tf.variable_scope(scope):
            n_l1 = 30
            w1_s = tf.get_variable('w1_s', [self.s_dim, n_l1], trainable=trainable)
            w1_a = tf.get_variable('w1_a', [self.a_dim, n_l1], trainable=trainable)
            b1 = tf.get_variable('b1', [1, n_l1], trainable=trainable)
            net = tf.nn.relu(tf.matmul(s, w1_s) + tf.matmul(a, w1_a) + b1)
            return tf.layers.dense(net, 1, trainable=trainable)  # Q(s,a)

env = gym.make(ENV_NAME)
env = env.unwrapped
env.seed(1)
s_dim = env.observation_space.shape[0]
a_dim = env.action_space.shape[0]
a_bound = env.action_space.high
ddpg = DDPG(a_dim, s_dim, a_bound)

var = 3  # control exploration
t1 = time.time()
for i in range(MAX_EPISODES):
    s = env.reset()
    ep_reward = 0
    for j in range(MAX_EP_STEPS): #没有明确停止条件的游戏都需要这么一个
        if RENDER:
            env.render()
        a = ddpg.choose_action(s)
        a = np.clip(np.random.normal(a, var),-2,2) #增加exploration noise 以actor输出的a为均值，var为方差进行选择a 同时保证a的值在【-2，2】
        s_, r, done, info = env.step(a)

        ddpg.store_transition(s, a, r/10, s_)
        if ddpg.pointer > MEMORY_CAPACITY: #存储的数据满了开始训练各个网络
            var *= 0.9995 #降低动作选择的随机性
            ddpg.learn() #超过10000才开始训练，每次从经验库中抽取一个batch，每走一步都会执行一次训练 单步更新
        s = s_
        ep_reward += r
        if j == MAX_EP_STEPS-1:
            print('Episode:', i, ' Reward: %i' % int(ep_reward), 'Explore: %.2f' % var, )
            # if ep_reward > -300:RENDER = True
            break
print('Running time: ', time.time() - t1)