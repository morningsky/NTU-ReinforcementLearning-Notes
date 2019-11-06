import tensorflow as tf
import numpy as np
import gym

'''
比较PG算法
PG loss = log_prob * v估计 （来自贝尔曼公式）
A2C loss = log_prob * TD-error（来自critic网络 表达当前动作的价值比平均动作的价值好多少）
DDPG ： critic不仅能影响actor actor也能影响critic 相当于critic不仅告诉actor的行为好不好，还告诉他应该怎么改进才能更好(传一个梯度 dq/da)
PPO: 对PG的更新加了限制，提高训练稳定性 相比于A2C 只是actor网络更加复杂
'''
class Actor(object): #本质还是policy gradient 不过A2C是单步更新
    def __init__(self, 
                 sess, #两个网络需要共用一个session 所以外部初始化
                 n_actions, 
                 n_features, 
                 lr=0.01, ):
        #self.ep_obs, self.ep_as, self.ep_rs =[],[],[] #由于是单步更新 所以不需要存储每个episode的数据
        self.sess = sess
        
        self.s = tf.placeholder(tf.float32, [1, n_features], "state")
        self.a = tf.placeholder(tf.int32, None, "act") #           
        self.td_error = tf.placeholder(tf.float32, None, "td_error")  # TD_error更新的幅度 td 的理解应该是 Q(s, a) - V(s), 某个动作价值减去平均动作价值
        
        with tf.variable_scope('Actor'): #将原来的name_scope换成variable_scope ，可以在一个scope里面共享变量
            l1 = tf.layers.dense(
                inputs=self.s,
                units=20,    # number of hidden units
                activation=tf.nn.relu,
                kernel_initializer=tf.random_normal_initializer(0., .1),    # weights
                bias_initializer=tf.constant_initializer(0.1),  # biases
                name='l1'
            )

            self.acts_prob = tf.layers.dense(
                inputs=l1,
                units=n_actions,    # output units
                activation=tf.nn.softmax,   # get action probabilities
                kernel_initializer=tf.random_normal_initializer(0., .1),  # weights
                bias_initializer=tf.constant_initializer(0.1),  # biases
                name='acts_prob'
            )
        
        #with tf.name_scope('loss'):
            # 最大化 总体 reward (log_p * R) 就是在最小化 -(log_p * R), 而 tf 的功能里只有最小化 loss
            #neg_log_prob = tf.reduce_sum(-tf.log(self.all_act_prob)*tf.one_hot(self.tf_acts, self.n_actions), axis=1) #加- 变为梯度下降
            #loss = tf.reduce_mean(neg_log_prob * self.tf_vt)
        with tf.variable_scope('loss'):
            log_prob = tf.log(self.acts_prob[0,self.a]) #[[0.1,0.2,0.3]] -> 0.1, if a=0
            self.loss = log_prob * self.td_error  # advantage (TD_error) guided loss

        with tf.name_scope('train'):
            self.train_op = tf.train.AdamOptimizer(lr).minimize(-self.loss)
            
    def choose_action(self, s): #选择行为
        s = s[np.newaxis, :]
        probs = self.sess.run(self.acts_prob, {self.s: s})   # get probabilities for all actions
        action = np.random.choice(np.arange(probs.shape[1]), p=probs.ravel()) 
        return action  # return a int
    

    def learn(self, s, a, td):
        s = s[np.newaxis, :]
        feed_dict = {self.s: s, self.a: a, self.td_error: td}
        _, loss = self.sess.run([self.train_op, self.loss], feed_dict)
        return loss


class Critic(object):
    def __init__(self, sess, n_features, lr=0.01, gamma=0.9):
        self.sess = sess

        self.s = tf.placeholder(tf.float32, [1, n_features], "state")
        self.v_ = tf.placeholder(tf.float32, [1, 1], "v_next")
        self.r = tf.placeholder(tf.float32, None, 'r')

        with tf.variable_scope('Critic'):
            l1 = tf.layers.dense(
                inputs=self.s,
                units=20,  # number of hidden units
                activation=tf.nn.relu,  # None
                # have to be linear to make sure the convergence of actor.
                # But linear approximator seems hardly learns the correct Q.
                kernel_initializer=tf.random_normal_initializer(0., .1),  # weights
                bias_initializer=tf.constant_initializer(0.1),  # biases
                name='l1'
            )

            self.v = tf.layers.dense(
                inputs=l1,
                units=1,  # output units
                activation=None,
                kernel_initializer=tf.random_normal_initializer(0., .1),  # weights
                bias_initializer=tf.constant_initializer(0.1),  # biases
                name='V'
            )

        with tf.variable_scope('squared_TD_error'):
            self.td_error = self.r + gamma * self.v_ - self.v
            self.loss = tf.square(self.td_error)    # TD_error = (r+gamma*V_next) - V_eval
        with tf.variable_scope('train'):
            self.train_op = tf.train.AdamOptimizer(lr).minimize(self.loss)

    def learn(self, s, r, s_):
        s, s_ = s[np.newaxis, :], s_[np.newaxis, :]

        v_ = self.sess.run(self.v, {self.s: s_})
        td_error, _ = self.sess.run([self.td_error, self.train_op],
                                        {self.s: s, self.v_: v_, self.r: r})
        return td_error

np.random.seed(2)
tf.set_random_seed(2)  # reproducible

# Superparameters
OUTPUT_GRAPH = False
MAX_EPISODE = 100#3000
DISPLAY_REWARD_THRESHOLD = 200  # renders environment if total episode reward is greater then this threshold
MAX_EP_STEPS = 1000   # maximum time step in one episode
RENDER = False  # rendering wastes time
GAMMA = 0.9     # reward discount in TD error
LR_A = 0.01    # learning rate for actor
LR_C = 0.05     # learning rate for critic

env = gym.make('CartPole-v0')
env.seed(1)  # reproducible
env = env.unwrapped

N_F = env.observation_space.shape[0]
N_A = env.action_space.n

from gym import Space

sess = tf.Session() #两个网络共用一个session

actor = Actor(sess, n_features=N_F, n_actions=N_A, lr=LR_A)
critic = Critic(sess, n_features=N_F, lr=LR_C)     # we need a good teacher, so the teacher should learn faster than the actor

sess.run(tf.global_variables_initializer())
        
if OUTPUT_GRAPH:
    tf.summary.FileWriter("logs/", sess.graph)
    
for i_episode in range(MAX_EPISODE):
    state = env.reset()
    t = 0
    r_list = []

    while True:
        if RENDER:
            env.render()
        action = actor.choose_action(state)
        state_, reward, done, info = env.step(action)
        if done: 
            reward=-20 #最后一步的奖励 一个trick
        r_list.append(reward)
        td_error = critic.learn(state, reward, state_)
        actor.learn(state, action, td_error)
        state = state_

        if done or t>= MAX_EP_STEPS:
            ep_rs_sum = sum(r_list)
            if 'running_reward' not in globals():
                running_reward = ep_rs_sum
            else:
                running_reward = running_reward * 0.95 + ep_rs_sum * 0.05
            if running_reward > DISPLAY_REWARD_THRESHOLD: RENDER = False  # rendering
            print("episode:", i_episode, "  reward:", int(running_reward))
            break


                

    

