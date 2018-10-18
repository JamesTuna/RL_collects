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

	def calculate_loss(self,ep_obs,ep_gs,back_prop=True):
		# calculate loss on ONE episode
		# ep_gs can be viewed as target/label
		ep_obs=checkAndConvert(ep_obs,self.name+' ep_obs')
		ep_gs=checkAndConvert(ep_gs,self.name+' ep_gs')
		# ep_obs: (T,n_features); ep_gs: (T,); predicted: (T,)
		predicted = self.net(ep_obs)
		loss = self.loss_func(predicted,ep_gs)
		if back_prop:
			loss.backward()
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

	def update_critor(self,ep_gs,ep_obs,train_iter):
		for i in range(train_iter):
			loss = self.critor.calculate_loss(ep_obs,ep_gs,back_prop=True)
			self.critor.update()

	def update_actor(self,ep_obs,ep_gs,ep_as,state_values,train_iter):
		for i in range(train_iter):
			loss = self.actor.calculate_loss(ep_obs,ep_gs,ep_as,state_values,back_prop=True)
			self.actor.update()

	def renew_actor(self):
		self.actor.renew_old_para()
		




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

if __name__ == '__main__':

	RENDER = False
	env = gym.make('CartPole-v0')
	env.seed(1)     # reproducible, general Policy gradient has high variance
	env = env.unwrapped
	agent = PPO(n_actions=env.action_space.n,n_features=env.observation_space.shape[0],
    			a_lr=0.001,c_lr=0.001,name='PPO_agent')


	for i_episode in range(MAX_EPISODES):

		epr = 0
		obs = env.reset()

		buffer_obs,buffer_as,buffer_rs,buffer_gs = [],[],[],[]

		for step in range(MAX_STEPS):

			if RENDER: env.render()
			act = agent.choose_action()
			buffer_obs.append(obs)
			buffer_as.append(act)
			obs,r,done,info = env.step(act)
			epr+=r
			buffer_rs.append(r)

			if step%BATCH == 0 or step==MAX_STEPS-1:
				gs = agent.get_v(obs)
				for r in reversed(buffer_rs):
					gs = r + GAMMA * gs
					buffer_gs.append(gs)
				buffer_gs = reversed(buffer_gs)
				# baseline, predicted by critor
				state_values = agent.critor.give_score(buffer_obs)
				# renew two nets
				agent.update_actor(buffer_obs,buffer_gs,buffer_as,state_values,train_iter=TRAIN_ITER)
				agent.update_critor(buffer_gs,buffer_obs,train_iter=TRAIN_ITER)
				agent.renew_actor()

				buffer_obs,buffer_as,buffer_rs,buffer_gs = [],[],[],[]







			




















