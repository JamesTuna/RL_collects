#! /usr/bin/env python3
import torch,math,pickle
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import gym as gym
import matplotlib.pyplot as plt


epsilon = 0.1
GAMMA = 0.99
BATCH = 256
MAX_STEPS = 10000
MAX_EPISODES = 5000
ACTOR_ITER = 10
CRITOR_ITER = 2
A_LR = 0.001
C_LR = 0.1

def checkAndConvert(x,msg):
	if type(x) != torch.Tensor:
		if type(x) == np.ndarray:
			return torch.Tensor(x)
		elif type(x) == list:
			return torch.Tensor(np.array(x))
		else:
			print("wrong at ",msg," ",type(x))
			exit()
	else:
		return x


class MLP_Net(nn.Module):

	def __init__(self,n_features,n_actions,use_softmax=True,name='MLP_net'):
		super(MLP_Net,self).__init__()
		self.name = name
		self.op = nn.Sequential(
			nn.Linear(n_features,32),
			nn.ReLU(),
			nn.Linear(32,32),
			nn.ReLU(),
			nn.Linear(32,n_actions)
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
			self.last_layer = self.op(x)
			return F.softmax(self.last_layer,dim=1)
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
		ep_gs=checkAndConvert(ep_gs,self.name+' ep_gs').view(-1,1)
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
		self.renew_old_para()

	def choose_action(self,observation):
		observation = checkAndConvert(observation,self.net.name+' choose action').view(1,-1)
		prob = self.net(observation).data.numpy().reshape(-1)
		#print('last_layer:\n',self.net.last_layer)
		act = np.random.choice(range(len(prob)),p=prob)
		#print(prob,act)
		return act

	def calculate_loss(self,ep_obs,ep_gs,ep_as,state_values,back_prop=True):
		# calculate loss on ONE trajectory
		# of form (to minimize) Σ - [π(a|s)/π'(a|s)] * A(a,s)
		# where A(a,s) is computed by gs - v(s), v(s) estimated by Critor
		ep_as = checkAndConvert(ep_as,self.net.name+' calculate_loss ep_as').long()
		ep_obs = checkAndConvert(ep_obs,self.net.name+' calculate_loss ep_obs')
		ep_gs = checkAndConvert(ep_gs,self.net.name+' calculate_loss ep_gs').view(-1,1)
		state_values = checkAndConvert(state_values,self.net.name+' calculate_loss state_values').view(-1,1)

		advantage = ep_gs - state_values
		# bad performance
		# advantage = (advantage-advantage.mean())/advantage.std()

		old_probs = self.old_net(ep_obs).detach() # [T,n_actions]
		new_probs = self.net(ep_obs) # [T,n_actions]

		'''
		# following code results in nan, why?
		ratio = new_probs/old_probs
		#ratio = torch.exp(new_probs-old_probs)
		# select probability of actions according to trajectory
		selected_ratio = ratio.gather(1,ep_as.view(-1,1))
		# clipped the ratio
		clipped_ratio = torch.clamp(selected_ratio,1-epsilon,1+epsilon)
		'''


		##################### new ############################
		old_probs = old_probs.gather(1,ep_as.view(-1,1))
		new_probs = new_probs.gather(1,ep_as.view(-1,1))
		selected_ratio = new_probs/old_probs
		clipped_ratio = torch.clamp(selected_ratio,1-epsilon,1+epsilon)
		#######################################################

		clipped_weighted_advan = clipped_ratio * advantage
		original_weighted_advan = selected_ratio * advantage
		final_weighted_advan = torch.min(clipped_weighted_advan,original_weighted_advan)
		loss = - torch.mean(final_weighted_advan)
		if back_prop:
			loss.backward()
		return loss

	def update(self):
		self.optimizer.step()
		self.optimizer.zero_grad()

	def renew_old_para(self):
		self.old_net.load_state_dict(self.net.state_dict())


class PPO(object):
	def __init__(self,n_features,n_actions,name,a_lr,c_lr):
		self.name = name
		self.actor = Actor(n_features,n_actions,name='_actor_1',learning_rate=a_lr)
		self.critor = Critor(n_features,name='_critor_1',learning_rate=c_lr)

	def choose_action(self,obs):
		return self.actor.choose_action(obs)

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

	def get_v(self,obs):
		return self.critor.give_score(obs).data.numpy()[0]

	def save_model(self,path):
		torch.save(self.actor.net.state_dict(),path+'_actor')
		torch.save(self.critor.net.state_dict(),path+'_critor')
		

if __name__ == '__main__':

	RENDER = False
	rewards = [[],[]]
	#env = gym.make('CartPole-v0')
	#env = gym.make('MountainCar-v0')
	env = gym.make('LunarLander-v2')
	env.seed(1)    
	env = env.unwrapped
	print(env.observation_space.shape)
	agent = PPO(n_actions=env.action_space.n,n_features=env.observation_space.shape[0],
    			a_lr=A_LR,c_lr=C_LR,name='PPO_agent')

	running_avg = None
	for i_episode in range(MAX_EPISODES):

		epr = 0
		obs = env.reset()

		buffer_obs,buffer_as,buffer_rs,buffer_gs = [],[],[],[]

		for step in range(1,MAX_STEPS+1):

			if RENDER: env.render()
			act = agent.choose_action(obs)
			buffer_obs.append(obs)
			buffer_as.append(act)
			obs,r,done,info = env.step(act)
			epr+=r
			buffer_rs.append(r)

			if step%BATCH == 0 or step==MAX_STEPS or done:
				if not done:
					gs = agent.get_v(obs)
				else:
					gs = 0
				for r in reversed(buffer_rs):
					gs = r + GAMMA * gs
					buffer_gs.append(gs)

				buffer_gs = list(reversed(buffer_gs))

				# baseline, predicted by critor
				state_values = agent.critor.give_score(buffer_obs).data.numpy()
				# renew two nets
				#print('actor update...')
				agent.update_actor(buffer_obs,buffer_gs,buffer_as,state_values,train_iter=ACTOR_ITER)
				#print('critor update...')
				agent.update_critor(buffer_gs,buffer_obs,train_iter=CRITOR_ITER)
				agent.renew_actor()

				buffer_obs,buffer_as,buffer_rs,buffer_gs = [],[],[],[]

			if done: break

		if i_episode == 0:
			running_avg = epr
		else:
			running_avg = 0.99*running_avg+0.01*epr
		'''
		# for cartPole
		if running_avg >=4999: 
			RENDER = True
			agent.save_model('./models/step'+str(step))
		'''
		print('epidode %s: '%(i_episode),epr,' runing_avg: ',running_avg)

		rewards[0].append(epr)
		rewards[1].append(running_avg)

		if RENDER: break
	# for mountainCar
	# agent.save_model('./models/mtcar_step'+str(running_avg))


	plt.plot([i for i in range(1,len(rewards[0])+1)],rewards[0],'r-')
	plt.plot([i for i in range(1,len(rewards[0])+1)],rewards[1],'b-')
	plt.xlabel('episodes')
	plt.ylabel('reward')
	plt.show()

	with open('lander.pkl','wb') as f:
		print('saving data...')
		pickle.dump(rewards,f)

	# for LunarLander-v2
	agent.save_model('./models/lander_rwd'+str(running_avg))











			




















