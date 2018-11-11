#! /usr/bin/env python3
from PPO import *

# 'CartPole-v0', 'MountainCar-v0'

GAME = 'LunarLander-v2'
#GAME = 'CartPole-v0'
#GAME = 'MountainCar-v0'
		

if __name__ == '__main__':

	RENDER = False
	rewards = [[],[]]

	env = gym.make(GAME)

	env.seed(1)    
	env = env.unwrapped 

	running_avg = None
	ep = 1
	while(True):
		epr = 0
		obs = env.reset()
		buffer_obs,buffer_as,buffer_rs,buffer_gs = [],[],[],[]
		while(True):
			env.render()
			act = env.action_space.sample()
			epr+=r
			if done:
				print('episode %s rewards: '%(ep),epr)
				if ep>1:
					running_avg = 0.99*running_avg+0.01*epr
				else:
					running_avg = epr
				break
		ep+=1











			




















