import tensorflow as tf
import numpy as np
import gym

class PolicyGradient:
    def __init__(self, 
                 n_actions, 
                 n_features, 
                 learning_rate=0.01, 
                 reward_decay=0.95, 
                 output_graph=False
    ):
        self.n_actions = n_actions
        self.n_features = n_features
        self.lr = learning_rate
        self.gamma = reward_decay
        self.ep_obs, self.ep_as, self.ep_rs =[],[],[] #states,actions,rewards
        self.__build_net()
        self.sess = tf.Session()
        
        if output_graph:
            # $ tensorboard --logdir=logs
            # http://0.0.0.0:6006/
            # tf.train.SummaryWriter soon be deprecated, use following
            tf.summary.FileWriter("logs/", self.sess.graph)
    
        self.sess.run(tf.global_variables_initializer())
        
    def __build_net(self): #PG网络
        with tf.name_scope('inputs'):
            self.tf_obs = tf.placeholder(tf.float32, [None, self.n_features],name="observations")
            self.tf_acts = tf.placeholder(tf.int32, [None,], name="actions_num")
            self.tf_vt = tf.placeholder(tf.float32, [None,], name="actions_value") #V(s,a)
            
            layer = tf.layers.dense(
                inputs = self.tf_obs,
                units = 10,
                activation = tf.nn.tanh,
                kernel_initializer=tf.random_normal_initializer(mean=0, stddev=0.3),
                bias_initializer=tf.constant_initializer(0.1),
                name = 'fc1'    
            )
            all_act = tf.layers.dense(
                inputs=layer,
                units=self.n_actions,   # 输出个数
                activation=None,    # 之后再加 Softmax
                kernel_initializer=tf.random_normal_initializer(mean=0, stddev=0.3),
                bias_initializer=tf.constant_initializer(0.1),
                name='fc2'
            )
            self.all_act_prob = tf.nn.softmax(all_act, name='act_prob')
            with tf.name_scope('loss'):
                # 最大化 总体 reward (log_p * R) 就是在最小化 -(log_p * R), 而 tf 的功能里只有最小化 loss
                neg_log_prob = tf.reduce_sum(-tf.log(self.all_act_prob)*tf.one_hot(self.tf_acts, self.n_actions), axis=1) #加- 变为梯度下降
                loss = tf.reduce_mean(neg_log_prob * self.tf_vt)

            with tf.name_scope('train'):
                self.train_op = tf.train.AdamOptimizer(self.lr).minimize(loss)
            
    def choose_action(self, observation): #选择行为
        prob_weights = self.sess.run(self.all_act_prob, feed_dict = {self.tf_obs: observation[np.newaxis, :]}) #[0,1,2]->[[0,1,2]] 所有action的概率 矩阵形式
        action = np.random.choice(range(prob_weights.shape[1]), p=prob_weights.ravel()) # 根据概率来选 action  range(prob_weights.shape[1]用0，1，2，表示动作
        return action
    
    def store_transition(self, s, a, r):#存储一个回合的经验
        self.ep_obs.append(s)
        self.ep_as.append(a)
        self.ep_rs.append(r)

    def learn(self):
        discounted_ep_rs_norm = self._discount_and_norm_rewards() # 衰减, 并标准化这回合的 reward
        self.sess.run(self.train_op, feed_dict={
            self.tf_obs: np.vstack(self.ep_obs),  # shape=[None, n_obs]
            self.tf_acts: np.array(self.ep_as),  # shape=[None, ]
            self.tf_vt: discounted_ep_rs_norm,  # shape=[None, ]
        })
        self.ep_obs, self.ep_as, self.ep_rs = [],[],[] #清空回合数据
        return discounted_ep_rs_norm # 返回这一回合的 state-action value


    def _discount_and_norm_rewards(self): #用bellman公式计算出vt(s,a)
        #discount
        discounted_ep_rs = np.zeros_like(self.ep_rs)
        running_add = 0
        for t in reversed(range(0, len(self.ep_rs))): #倒数遍历这个episode中的reward
            running_add = running_add * self.gamma + self.ep_rs[t]
            discounted_ep_rs[t] = running_add
            # r1,r2,r3 -> r1+r2*gamma+r3*gamma^2, r2+r3*gamma, r3

        #normalize
        discounted_ep_rs -= np.mean(discounted_ep_rs)
        discounted_ep_rs /= np.std(discounted_ep_rs)
        return discounted_ep_rs



#将算法应用起来吧！
RENDER = False  # 在屏幕上显示模拟窗口会拖慢运行速度, 我们等计算机学得差不多了再显示模拟
DISPLAY_REWARD_THRESHOLD = 1000  # 当 回合总 reward 大于 400 时显示模拟窗口

#env = gym.make('CartPole-v0')   # CartPole 2个动作 向左 向右
env = gym.make('MountainCar-v0') #3个动作 左侧加速、不加速、右侧加速

env = env.unwrapped     # 取消限制
env.seed(1)     # 普通的 Policy gradient 方法, 使得回合的 variance 比较大, 所以我们选了一个好点的随机种子

print(env.action_space)     # 显示可用 action 
print(env.observation_space)    # 显示可用 state 的 observation
print(env.observation_space.high)   # 显示 observation 最高值
print(env.observation_space.low)    # 显示 observation 最低值

# 定义
RL = PolicyGradient(
    n_actions=env.action_space.n,
    n_features=env.observation_space.shape[0],
    learning_rate=0.02,
    reward_decay=0.99,   # gamma
    # output_graph=True,    # 输出 tensorboard 文件
)

for i_episode in range(100):
    observation = env.reset()
    while True:
        if RENDER:
            env.render()
        action = RL.choose_action(observation)
        observation_, reward, done, info = env.step(action)
        RL.store_transition(observation, action, reward)
        
        if done:
            ep_rs_sum = sum(RL.ep_rs)
            if 'running_reward' not in globals():
                running_reward = ep_rs_sum
            else:
                running_reward = running_reward *0.99 + ep_rs_sum *0.01 #不是简单的求和展示当下rewad 比较科学
            print("episode:", i_episode, "reward:", int(running_reward))
            vt = RL.learn() #学习 输出vt 
            break
    
        observation =  observation_


