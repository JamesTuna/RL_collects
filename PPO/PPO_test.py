#! /usr/bin/env python3
from PPO import *

epsilon = 0.1
GAMMA = 0.99
BATCH = 256
MAX_STEPS = 10000
MAX_EPISODES = 5000
ACTOR_ITER = 10
CRITOR_ITER = 2
A_LR = 0.001
C_LR = 0.1

		

if __name__ == '__main__':

	RENDER = False
	rewards = [[],[]]
	#env = gym.make('CartPole-v0')
	#env = gym.make('MountainCar-v0')
	env = gym.make('LunarLander-v2')
	env.seed(1)    
	env = env.unwrapped
	agent = PPO(n_actions=env.action_space.n,n_features=env.observation_space.shape[0],
    			a_lr=A_LR,c_lr=C_LR,name='PPO_agent')
	agent.actor.net.load_state_dict(torch.load('./models/lander_rwd236.19130072672505_actor'))
	agent.critor.net.load_state_dict(torch.load('./models/lander_rwd236.19130072672505_critor'))
	running_avg = None
	ep = 1
	while(True):
		epr = 0
		obs = env.reset()
		buffer_obs,buffer_as,buffer_rs,buffer_gs = [],[],[],[]
		while(True):
			env.render()
			act = agent.choose_action(obs)
			buffer_obs.append(obs)
			buffer_as.append(act)
			obs,r,done,info = env.step(act)
			epr+=r
			buffer_rs.append(r)
			if done:
				print('episode %s rewards: '%(ep),epr)
				if ep>1:
					running_avg = 0.99*running_avg+0.01*epr
				else:
					running_avg = epr
				buffer_obs,buffer_as,buffer_rs,buffer_gs = [],[],[],[]
				break
		ep+=1











			




















