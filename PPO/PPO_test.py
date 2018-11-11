#! /usr/bin/env python3
from PPO import *

# 'CartPole-v0', 'MountainCar-v0'

GAME = 'LunarLander-v2'
#GAME = 'CartPole-v0'
#GAME = 'MountainCar-v0'
#GAME = 'Pendulum-v0'
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

	env = gym.make(GAME)

	env.seed(1)    
	env = env.unwrapped
	if GAME == 'Pendulum-v0':
		ACTION_SPACE = 25
		agent = PPO(n_actions=ACTION_SPACE,n_features=env.observation_space.shape[0],
    			a_lr=A_LR,c_lr=C_LR,name='PPO_agent')
	else:
		agent = PPO(n_actions=env.action_space.n,n_features=env.observation_space.shape[0],
    			a_lr=A_LR,c_lr=C_LR,name='PPO_agent')
	
	if GAME == 'LunarLander-v2':
		# for lunarlander demo
		agent.actor.net.load_state_dict(torch.load('./models/lander_rwd236.19130072672505_actor'))
		agent.critor.net.load_state_dict(torch.load('./models/lander_rwd236.19130072672505_critor'))
	elif GAME == 'CartPole-v0':
		# for cartPole demo
		agent.actor.net.load_state_dict(torch.load('./models/step5000_actor'))
		agent.critor.net.load_state_dict(torch.load('./models/step5000_critor'))
	elif GAME == 'MountainCar-v0':
		# for mountain car demo
		agent.actor.net.load_state_dict(torch.load('./models/mtcar_step-116.1643907627642_actor'))
		agent.critor.net.load_state_dict(torch.load('./models/mtcar_step-116.1643907627642_critor'))
	elif GAME == 'Pendulum-v0':
		agent.actor.net.load_state_dict(torch.load('./models/pendulumn_rwd-892.8866369093198_actor'))
		agent.critor.net.load_state_dict(torch.load('./models/pendulumn_rwd-892.8866369093198_critor'))
	else:
		print('game unknown')
		exit()

	running_avg = None
	ep = 1
	while(True):
		epr = 0
		obs = env.reset()
		buffer_obs,buffer_as,buffer_rs,buffer_gs = [],[],[],[]
		while(True):
			env.render()
			act = agent.choose_action(obs)
			if GAME == 'Pendulum-v0':
				f_action = (act-(ACTION_SPACE-1)/2)/((ACTION_SPACE-1)/4)   # [-2 ~ 2] float actions
				f_action = np.array([f_action])

			if GAME == 'Pendulum-v0':
				obs,r,done,info = env.step(f_action)
			else:
				obs,r,done,info = env.step(act)

			buffer_obs.append(obs)
			buffer_as.append(act)
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











			




















