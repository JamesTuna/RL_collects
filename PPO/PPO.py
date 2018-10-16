#! /usr/bin/env python3
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

epsilon = 0.2
GAMMA = 1

def checkAndConvert(x,msg):
	'''
	for debug
	print('check ',msg)
	'''
	if type(x) != torch.Tensor:
		if type(x) == np.ndarray:
			return torch.Tensor(x)
		elif type(x) == list:
			return torch.Tensor(np.array(x))
		else:
			print("wrong at ",msg)
			exit()
	else:
		return x


class MLP_Net(nn.Module):

	def __init__(self,n_features,n_actions,use_softmax=True,name='MLP_net'):
		super(MLP_Net,self).__init__()
		self.name = name
		self.op = nn.Sequential(
			nn.Linear(n_features,10),
			nn.ReLU(),
			nn.Linear(10,10),
			nn.ReLU(),
			nn.Linear(10,n_actions)
		)
		self.init_weights()
		self.use_softmax = use_softmax

	def init_weights(self):
		for m in self.modules():
			if type(m) == nn.Linear:
				nn.init.normal_(m.weight,mean=0,std=0.3)
				nn.init.constant_(m.bias,0.1)

	def forward(self,x):
		#assert type(x) == torch.Tensor, 'input of net %s is not Tensor!'%(self.name)
		x = checkAndConvert(x,self.name+' forward')
		if self.use_softmax:
			return F.softmax(self.op(x),dim=1)
		else:
			return self.op(x)

class Critor(object):

	def __init__(self,n_features,name,learning_rate):
		self.net = MLP_Net(n_features,1,use_softmax=False,name=name)
		self.loss_func = nn.MSELoss(size_average=True)
		self.optimizer = torch.optim.Adam(self.net.parameters(),lr=learning_rate)
		self.loss = None
		self.name = name

	def calculate_loss(self,ep_obs,ep_rs,ep_gs,back_prop=True):
		#calculate loss on ONE episode
		#assert type(ep_obs) == torch.Tensor, 'ep_obs of net %s is not Tensor!'%(self.net.name)
		#assert type(ep_rs) == torch.Tensor, 'ep_rs of net %s is not Tensor!'%(self.net.name)
		ep_obs=checkAndConvert(ep_obs,self.name+' ep_obs')
		ep_rs=checkAndConvert(ep_rs,self.name+' ep_rs')
		ep_gs=checkAndConvert(ep_gs,self.name+' ep_gs')
		# ep_obs: (T,n_features); ep_rs: (T,); predict: (T,)
		predicted = self.net(ep_obs)
		next_state = [ep_gs[i] for i in range(1,len(ep_gs))]
		next_state.append(0)
		next_state = torch.Tensor(np.array(next_state)).view(-1,1)
		# predicted: [r1,r2,r3...,rt]
		# next_stat: [g2,g3,g4...,0]
		target = ep_rs + GAMMA * next_state
		loss = self.loss_func(predicted,target)
		if back_prop:
			loss.backward()
		'''
		# for debug
		print('ep_obs\n',ep_obs)
		print('ep_rs\n',ep_rs)
		print('ep_gs\n',ep_gs)
		print('predicted\n',predicted)
		print('next_state\n',next_state)
		print('target\n',target)
		print('loss\n',loss)
		print('grad\n')
		for para in self.net.parameters():
			print(para.grad)
		'''
		return loss

	def update(self):
		self.optimizer.step()
		self.optimizer.zero_grad()

	def give_score(self,ep_obs):
		ep_obs=checkAndConvert(ep_obs,self.name+' ep_obs')
		predicted = self.net(ep_obs)
		return predicted

class Actor(object):
	def __init__(self,n_features,n_actions,name,learning_rate):
		self.net = MLP_Net(n_features,n_actions,name=name+'_new')
		self.optimizer = torch.optim.Adam(self.net.parameters(),lr=learning_rate)
		self.old_net = MLP_Net(n_features,n_actions,name=name+'_old')

	def choose_action(self,observation):
		observation = checkAndConvert(observation,self.net.name+' choose action').view(1,-1)
		prob = self.net(observation).data.numpy().reshape(-1)
		return np.random.choice(range(len(prob)),p=prob)

	def calculate_loss(self,ep_obs,ep_gs,ep_as,state_values,back_prop=True):
		# calculate loss on ONE trajectory
		# of form (to minimize) Σ - [π(a|s)/π'(a|s)] * A(a,s)
		# where A(a,s) is computed by gs - v(s), v(s) estimated by Critor
		ep_as = checkAndConvert(ep_as,self.net.name+' calculate_loss ep_as').long()
		ep_obs = checkAndConvert(ep_obs,self.net.name+' calculate_loss ep_obs')
		ep_gs = checkAndConvert(ep_gs,self.net.name+' calculate_loss ep_gs')
		state_values = checkAndConvert(state_values,self.net.name+' calculate_loss state_values')

		advantage = ep_gs - state_values

		old_probs = self.old_net(ep_obs).detach() # [T,n_actions]
		new_probs = self.net(ep_obs) # [T,n_actions]
		ratio = new_probs/old_probs
		# select probability of actions according to trajectory
		selected_ratio = ratio.gather(1,ep_as.view(-1,1))
		# clipped the ratio
		clipped_ratio = torch.clamp(selected_ratio,1-epsilon,1+epsilon)
		
		clipped_weighted_advan = clipped_ratio * advantage
		original_weighted_advan = selected_ratio * advantage
		final_weighted_advan = torch.min(clipped_weighted_advan,original_weighted_advan)
		loss = -torch.sum(final_weighted_advan)
		if back_prop:
			loss.backward()
		'''
		# for debug
		print('advan: \n',advantage)
		print('prob_ratio: \n',ratio)
		print('acts: \n',ep_as)
		print('selected_ratio, \n',selected_ratio)
		print('clipped_ratio with ',epsilon,'\n',clipped_ratio)
		print('original_weighted_advan:\n',original_weighted_advan)
		print('clipped_weighted_advan:\n',clipped_weighted_advan)
		print('final_weighted_advan:\n',final_weighted_advan)
		print('loss: \n',loss)
		#loss.backward()
		print('after backward, grad')
		for para in self.net.parameters():
			print('new para ',para.grad)
		for para in self.old_net.parameters():
			print('old para ',para.grad)
		'''
		return loss

	def update(self):
		self.optimizer.step()
		self.optimizer.zero_grad()

	def renew_old_para(self):
		print('renew old parameters for %s'%(self.old_net.name))
		self.old_net.load_state_dict(self.net.state_dict())


class PPO(object):
	def __init__(self,n_features,n_actions,name,a_lr,c_lr):
		self.name = name
		self.actor = Actor(n_features,n_actions,name='_actor_1',learning_rate=a_lr)
		self.critor = Critor(n_features,name='_critor_1',learning_rate=c_lr)

	def update(self,ep_obs,ep_as,ep_rs):
		# preprocess to calculate discounted reawrds
		# last one of ep_rs
		for i in range
		# update critor
		state_values = critor.give_score(ep_obs)




'''
# Actor debug
a = Actor(4,2,'actor',0.001)
#a.calculate_loss(ep_obs=[[0,0,0,1],[0,1,0,0]],ep_gs=[[1.5],[1.5]],ep_as=[1,1],state_values=[[0],[1]])
print(a.choose_action([1,0,0,0]))
'''
'''
# Critor debug
c = Critor(4,'critor',0.001)
c.calculate_loss(ep_obs=[[0,0,0,1],[0,1,0,0]],ep_gs=[[2],[1]],ep_rs=[[1],[1]])
'''
