#pendulum
#动作空间：只有一维力矩 长度为1
#状态空间：一维速度，长度为3

'''
    Critic网络直接给出V(s)
    Actor网络由2部分组成 oldpi pi
    PPO升级于A2C（critic按batch更新，离线训练，有2个pi），升级于PG（加入critic网络，利用advantage引导pg优化）
'''
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import gym

EP_MAX = 1000
EP_LEN = 200
GAMMA = 0.9
A_LR = 0.0001
C_LR = 0.0002
BATCH = 32
A_UPDATE_STEPS = 10
C_UPDATE_STEPS = 10
S_DIM, A_DIM = 3, 1 #pendulum游戏
METHOD = [
    dict(name='kl_pen', kl_target=0.01, lam=0.5),   # KL penalty
    dict(name='clip', epsilon=0.2),                 # Clipped surrogate objective, find this is better
][1]        # choose the method for optimization



class PPO(object):
    def __init__(self):
        self.sess = tf.Session()
        self.tfs = tf.placeholder(tf.float32, [None, S_DIM], 'state') #[N_each_batch,DIM]
        
        #搭建AC网络 critic
        with tf.variable_scope('critic'):
            layer1 = tf.layers.dense(self.tfs, 100, tf.nn.relu)
            self.v = tf.layers.dense(layer1, 1)
            self.tfdc_r = tf.placeholder(tf.float32, [None, 1], 'discounted_r')
            self.advantage = self.tfdc_r - self.v # discounted reward - Critic 出来的 state value
            self.closs = tf.reduce_mean(tf.square(self.advantage)) # mse loss of critic
            self.ctrain_op = tf.train.AdamOptimizer(C_LR).minimize(self.closs)

        pi, pi_params = self._build_anet('pi', trainable=True)
        oldpi, oldpi_params = self._build_anet('oldpi', trainable=False) #每个pi的本质是一个概率分布
        with tf.variable_scope('sample_action'):
            self.sample_op = tf.squeeze(pi.sample(1), axis=0) #按概率分布pi选择一个action
        with tf.variable_scope('update_oldpi'):
            self.update_oldpi_op = [oldp.assign(p) for p, oldp in zip(pi_params, oldpi_params)] #将pi的参数复制给oldpi

        self.tfa = tf.placeholder(tf.float32, [None, A_DIM], 'action')
        self.tfadv = tf.placeholder(tf.float32, [None, 1], 'advantage')
        with tf.variable_scope('loss'):
            with tf.variable_scope('surrogate'): 
                ratio  = pi.prob(self.tfa) / oldpi.prob(self.tfa) #(New Policy/Old Policy) 的比例
                surr = ratio * self.tfadv #surrogate objective
            if METHOD['name'] == 'kl_pen':  # 如果用 KL penatily
                self.tflam = tf.placeholder(tf.float32, None, 'lambda')
                kl = tf.distributions.kl_divergence(oldpi, pi)
                self.kl_mean = tf.reduce_mean(kl)
                self.aloss = tf.reduce_mean(surr - self.tflam * kl) #actor 最终的loss function
            else:   # 如果用 clipping 的方式
                self.aloss = tf.reduce_mean(tf.minimum(surr, tf.clip_by_value(ratio, 1-METHOD['epsilon'], 1+METHOD['epsilon'])*self.tfadv))
        
        with tf.variable_scope('atrain'):
            self.atrain_op = tf.train.AdamOptimizer(A_LR).minimize(-self.aloss)
        
        self.sess.run(tf.global_variables_initializer())


    def update(self, s, a, r): #update ppo
        # 先要将 oldpi 里的参数更新 pi 中的
        self.sess.run(self.update_oldpi_op) 
        adv = self.sess.run(self.advantage, {self.tfs:s, self.tfdc_r:r})
        # adv = (adv - adv.mean())/(adv.std()+1e-6)     # sometimes helpful
        # update actor
        # 更新 Actor 时, kl penalty 和 clipping 方式是不同的
        if METHOD['name'] == 'kl_pen':
            for _ in range(A_UPDATE_STEPS): #actor 一次训练更新10次
                _, kl = self.sess.run(
                    [self.atrain_op, self.kl_mean],
                    {self.tfs: s, self.tfa: a, self.tfadv: adv, self.tflam: METHOD['lam']})
                if kl > 4*METHOD['kl_target']:  # this in in google's paper
                    break
            if kl < METHOD['kl_target'] / 1.5:  # adaptive lambda, this is in OpenAI's paper
                METHOD['lam'] /= 2
            elif kl > METHOD['kl_target'] * 1.5:
                METHOD['lam'] *= 2
            METHOD['lam'] = np.clip(METHOD['lam'], 1e-4, 10)    # sometimes explode, this clipping is my solution
        else:   # clipping method, find this is better (OpenAI's paper)
            [self.sess.run(self.atrain_op, {self.tfs: s, self.tfa: a, self.tfadv: adv}) for _ in range(A_UPDATE_STEPS)] #actor 一次训练更新10次
        # 更新 Critic 的时候, 他们是一样的  critic一次训练更新10次
        [self.sess.run(self.ctrain_op, {self.tfs: s, self.tfdc_r: r}) for _ in range(C_UPDATE_STEPS)]

    def choose_action(self, s):
        s = s[np.newaxis, :]
        a = self.sess.run(self.sample_op, {self.tfs:s})[0]
        return np.clip(a,-2,2) #动作不要超出【-2，2】的范围 因为是按概率分布取动作 所以加上这一步很有必要！
    
    def get_v(self, s): #V(s)状态值 由critic网络给出
        if s.ndim < 2: 
            s = s[np.newaxis, :]
        return self.sess.run(self.v, {self.tfs:s})[0,0]

    def _build_anet(self, name, trainable): #critic网络输出动作的概率分布 包含参数均值u与方差sigma
        with tf.variable_scope(name):
            l1 = tf.layers.dense(self.tfs, 100, tf.nn.relu, trainable=trainable)
            mu = 2 * tf.layers.dense(l1, A_DIM, tf.nn.tanh, trainable=trainable)
            sigma = tf.layers.dense(l1, A_DIM, tf.nn.softplus, trainable=trainable)
            norm_dist = tf.distributions.Normal(loc=mu, scale=sigma)
        params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=name)
        return norm_dist, params

env = gym.make('Pendulum-v0').unwrapped

ppo = PPO()
all_ep_r = []


for ep in range(EP_MAX):
    s = env.reset()
    buffer_s, buffer_a, buffer_r = [],[],[]
    ep_r = 0
    for t in range(EP_LEN):
        env.render()
        a = ppo.choose_action(s)
        s_, r, done, info = env.step(a)
        buffer_s.append(s)
        buffer_a.append(a)
        buffer_r.append((r+8)/8) # normalize reward, 发现有帮助
        s = s_
        ep_r += r #一个episode的reward之和

        # 如果 buffer 收集一个 batch 了或者 episode 完了
        #则更新ppo
        if (t+1) % BATCH == 0 or t == EP_LEN -1:
            #计算折扣奖励
            v_s_ = ppo.get_v(s_)
            discounted_r = []
            for r in buffer_r[::-1]:
                v_s_ = r + GAMMA * v_s_
                discounted_r.append(v_s_)
            discounted_r.reverse()

            bs, ba, br = np.vstack(buffer_s), np.vstack(buffer_a), np.array(discounted_r)[:, np.newaxis] #存入一个Batch
            #清空buffer
            buffer_s, buffer_a, buffer_r = [],[],[]
            ppo.update(bs, ba, br) #训练PPO
        if ep == 0:
            all_ep_r.append(ep_r)
        else:
            all_ep_r.append(all_ep_r[-1]*0.9 + ep_r*0.1)

        print('Ep: %i' % ep,"|Ep_r: %i" % ep_r,("|Lam: %.4f" % METHOD['lam']) if METHOD['name'] == 'kl_pen' else '',)


#plt.plot(np.arange(len(all_ep_r)), all_ep_r)
#plt.xlabel('Episode')
#plt.ylabel('Moving averaged episode reward')
#plt.show()





