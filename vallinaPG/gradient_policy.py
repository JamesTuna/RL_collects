#! /usr/bin/env python3
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F 

LR = 0.01
N_FEATURES = 10
N_ACTIONS = 2
BATCH_SIZE = 32

class Net(nn.Module):
	def __init__(self,n_features,n_actions):
		# network output probability of different action
		# at any given state, with n_features
		super(Net,self).__init__()
		self.op = nn.Sequential(
			nn.Linear(n_features,10),
			nn.ReLU(),
			nn.Linear(10,10),
			nn.ReLU(),
			nn.Linear(10,n_actions),
		)
		self.init_weight()

	def init_weight(self):
		for m in self.modules():
			if type(m) == nn.Linear:
				nn.init.normal_(m.weight,mean=0.,std=0.3)
				nn.init.constant_(m.bias,0.1)
				#nn.init.constant_(m.weight,1)

	def forward(self,x):
		ll = self.op(x)
		return F.softmax(ll,dim=1)


class Actor(object):
	def __init__(self,n_features,n_actions,learning_rate,reward_decay=0.95):
		self.n_actions = n_actions
		self.n_features = n_features
		self.lr = learning_rate
		self.net = Net(n_features,n_actions)
		self.gamma = reward_decay
		self.ep_obs, self.ep_as, self.ep_rs = [], [], []
		self.learning_curve = []
		self.N = 1
		self.optimizer = torch.optim.Adam(self.net.parameters(), lr=learning_rate)
		self.cashe_episode = 0
		self.moving_rewards = 0

	def save_model(self,path):
		torch.save(self.net.state_dict(),path)

	def store_transition(self, s, a, r):
		self.ep_obs.append(s)
		self.ep_as.append(a)
		self.ep_rs.append(r)

	# state of shape [1,n_features]
	def choose_action(self,state):
		state = np.expand_dims(np.array(state),0)
		state = torch.Tensor(state)
		# net_op shape [1,actions]
		net_op = self.net(state)
		prob = net_op.data.numpy()[0]
		action = np.random.choice(range(self.n_actions), p=prob)  # select action w.r.t the actions prob
		return action

	def learn_action(self,state,action,reward):
		# stepwise updating
		# update parameter proportional to reward * gradient(log(P(a|θ)))
		para_list = list(self.net.parameters())
		# calculate gradient
		torch.log(self.net(state)[0][action]).backward()
		'''
		# manually gradient
		for para in para_list:
			grad = para.grad.data.numpy()
			para.data.add_(self.lr*reward*grad)
			para.grad.data.zero_()
		'''
		loss = -torch.log(self.net(state)[0][action])
		self.optimizer.zero_grad()
		loss.backward()
		for para in para_list:
			para.grad *= reward
		self.optimizer.step()


	def learn_episode(self):
		print("learning episode, total %s moves"%len(self.ep_obs))
		discounted_ep_rs_norm = self._discount_and_norm_rewards()
		rewards = np.reshape(discounted_ep_rs_norm,(len(discounted_ep_rs_norm),1))
		# shape [n,n_features]
		states = np.array(self.ep_obs)
		# [n,n_actions]
		predicts = self.net(torch.Tensor(states))
		# [n,1]
		rewards = torch.Tensor(rewards)
		# loglikelihood
		predicts = torch.log(predicts)
		# weighted by rewards
		predicts = predicts * rewards
		labels =  torch.Tensor(np.array(self.ep_as)).long()
		loss_func = nn.NLLLoss()
		loss = loss_func(predicts,labels)
		# built in optimizer
		
		loss.backward()
		self.cashe_episode += 1
		if self.cashe_episode >= self.N:
			self.optimizer.step()
			self.optimizer.zero_grad()
			self.cashe_episode = 0
		self.ep_obs, self.ep_as, self.ep_rs = [], [], []
		'''
		
		# manually update parameters
		loss.backward()
		for para in self.net.parameters():
			grad = para.grad
			para.data.add_(-20*self.lr*grad)
			para.grad.data.zero_()
		self.ep_obs, self.ep_as, self.ep_rs = [], [], []
		'''
		return discounted_ep_rs_norm

	def _discount_and_norm_rewards(self):
		# discount episode rewards
		discounted_ep_rs = np.zeros_like(self.ep_rs)
		running_add = 0
		for t in reversed(range(0, len(self.ep_rs))):
			running_add = running_add * self.gamma + self.ep_rs[t]
			discounted_ep_rs[t] = running_add
		# normalize episode rewards
		discounted_ep_rs -= np.mean(discounted_ep_rs)
		discounted_ep_rs /= np.std(discounted_ep_rs)
		
		# valina policy gradient
		# use moving average as baseline
		'''
		running_add /= 1
		self.moving_rewards = 0.9*self.moving_rewards + 0.1 * running_add
		for t in range(len(self.ep_rs)):
			discounted_ep_rs[t] = running_add - self.moving_rewards
		'''
		return discounted_ep_rs

	def learn(self):
		# batch gradient
		a = self.learn_episode()
		self.ep_obs, self.ep_as, self.ep_rs = [], [], []
		return a
		'''
		# stepwise updating
		print("learning episode, total %s moves"%len(self.ep_obs))
		discounted_ep_rs_norm = self._discount_and_norm_rewards()
		self.ep_obs = np.expand_dims(np.array(self.ep_obs),axis=1)
		for i in range(len(self.ep_obs)):
			state = torch.Tensor(self.ep_obs[i])
			action = self.ep_as[i]
			reward = discounted_ep_rs_norm[i]
			self.learn_action(state,action,reward)
		self.ep_obs, self.ep_as, self.ep_rs = [], [], []
		return discounted_ep_rs_norm
		'''

if __name__ == "__main__":
	actor = Actor(N_FEATURES,N_ACTIONS,LR)
	states = np.random.randn(5,N_FEATURES)
	actions = np.random.choice(N_ACTIONS,5)
	rewards = np.random.randn(5,1)
	actor.learn_episode(states,actions,rewards)



	    