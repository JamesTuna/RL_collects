#! /usr/bin/env python3

import gym

env = gym.make('MountainCar-v0')
env.reset()
#env = env.unwrapped

def random_game():
	counter=0
	reward=0
	global env
	while(True):
		action = env.action_space.sample()
		obs,r,done,info = env.step(action)
		reward+=r
		if done:
			print('done') 
			break
		counter+=1
	print(counter,reward)

random_game()
