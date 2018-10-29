#! /usr/bin/env python3
import torch
import torch.nn as nn
import numpy as np
import gym

env = gym.make('CartPole-v0')
env = env.unwrapped

def init_weight(m):
    if type(m) == nn.Linear:
        print('Linear',m)
        nn.init.normal_(m.weight,mean=0.,std=0.3)
        nn.init.constant_(m.bias,0.1)

class Net(nn.Module):
    def __init__(self,n_features,n_actions):
        super(Net,self).__init__()
        self.op = nn.Sequential(
            nn.Linear(n_features,10),
            nn.ReLU(),
            nn.Linear(10,n_actions),
        )
        for m in self.modules():
            m.apply(init_weight)

    def forward(self,x):
        return self.op(x)

class DQN(object):
    def __init__(
            self,
            n_actions,
            n_features,
            learning_rate=0.01,
            reward_decay=0.9,
            e_greedy=0.9,
            replace_target_iter=300,
            memory_size=500,
            batch_size=32,
            e_greedy_increment=None,
            output_graph=False,
    ):
        # build two nets
        self.n_actions = n_actions
        self.n_features = n_features
        self.lr = learning_rate
        self.gamma = reward_decay
        self.epsilon_max = e_greedy
        self.replace_target_iter = replace_target_iter
        self.memory_size = memory_size
        self.batch_size = batch_size
        self.epsilon_increment = e_greedy_increment
        self.epsilon = 0 if e_greedy_increment is not None else self.epsilon_max

        self.eval_net,self.target_net = Net(n_features,n_actions),Net(n_features,n_actions)
        self.learn_step_counter = 0
        self.memory_counter = 0
        self.memory = np.zeros((memory_size,n_features*2+2))
        self.optimizer = torch.optim.Adam(self.eval_net.parameters(), lr=learning_rate)
        self.loss_func = nn.MSELoss()


    def choose_action(self,x):
        # choose action based on obeservation
        x = torch.unsqueeze(torch.FloatTensor(x),0)
        if np.random.uniform() < self.epsilon:
            action_value = self.eval_net.forward(x)
            action = torch.max(action_value,1)[1].data.numpy()
            action = action[0]
        else:
            action = np.random.randint(0,self.n_actions)
        return action

    def store_transition(self,s,a,r,s_):
        transition = np.hstack((s,a,r,s_))
        index = self.memory_counter%self.memory_size
        self.memory[index,:] = transition
        self.memory_counter += 1

    def learn(self):
        if self.learn_step_counter % self.replace_target_iter == 0:
            print('\ntarget_params_replaced\n')
            self.target_net.load_state_dict(self.eval_net.state_dict())

        if self.memory_counter >= self.memory_size:
            sample_index = np.random.choice(self.memory_size, self.batch_size)
        else:
            sample_index = np.random.choice(self.memory_counter,self.batch_size)

        b_memory = self.memory[sample_index,:]
        b_s = torch.FloatTensor(b_memory[:,:self.n_features])
        b_a = torch.LongTensor(b_memory[:,self.n_features:self.n_features+1].astype(int))
        b_r = torch.FloatTensor(b_memory[:, self.n_features+1:self.n_features+2])
        b_s_ = torch.FloatTensor(b_memory[:, -self.n_features:])

        q_eval = self.eval_net(b_s).gather(1, b_a)
        q_next = self.target_net(b_s_).detach()
        q_target = b_r + self.gamma * q_next.max(1)[0].view(self.batch_size, 1)
        loss = self.loss_func(q_eval, q_target)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.epsilon = self.epsilon + self.epsilon_increment if self.epsilon < self.epsilon_max else self.epsilon_max
        self.learn_step_counter += 1

dqn = DQN(  n_actions=env.action_space.n,
            n_features=env.observation_space.shape[0],
            learning_rate=0.01, e_greedy=0.9,
            replace_target_iter=100, memory_size=2000,
            e_greedy_increment=0.001,)

for episode in range(400):
    s = env.reset()
    ep_r = 0
    while True:
        env.render()
        a = dqn.choose_action(s)
        s_, r, done, info = env.step(a)
        x, x_dot, theta, theta_dot = s_
        r1 = (env.x_threshold - abs(x)) / env.x_threshold - 0.8
        r2 = (env.theta_threshold_radians - abs(theta)) / env.theta_threshold_radians - 0.5
        r = r1 + r2
        dqn.store_transition(s,a,r,s_)
        ep_r += r
        if dqn.memory_counter >= dqn.memory_size:
            dqn.learn()
            if done:
                print('EP: ',episode,'| Ep_r: ',round(ep_r,2),' |epsilon: ',dqn.epsilon)
        if done:
            break
        s = s_
