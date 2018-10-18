#! /usr/bin/env python3
import gym
#from RL_brain import PolicyGradient
from gradient_policy import Actor
import matplotlib.pyplot as plt
import pickle

rewards = [[],[]]
GAMMA = 0.99

DISPLAY_REWARD_THRESHOLD = 500  # renders environment if total episode reward is greater then this threshold
RENDER = False  # rendering wastes time

env = gym.make('CartPole-v0')
#env = gym.make('LunarLander-v2')
env.seed(1)     # reproducible, general Policy gradient has high variance
env = env.unwrapped

print(env.action_space)
print(env.observation_space)
print(env.observation_space.high)
print(env.observation_space.low)
'''
RL = PolicyGradient(
    n_actions=env.action_space.n,
    n_features=env.observation_space.shape[0],
    learning_rate=0.02,
    reward_decay=0.99,
    # output_graph=True,
)

'''
RL = Actor(
    n_actions=env.action_space.n,
    n_features=env.observation_space.shape[0],
    learning_rate=0.01,
    reward_decay=GAMMA,

)

for i_episode in range(1000):

    observation = env.reset()

    while True:
        if RENDER: env.render()

        action = RL.choose_action(observation)

        observation_, reward, done, info = env.step(action)

        RL.store_transition(observation, action, reward)

        if done:
            ep_rs_sum = sum(RL.ep_rs)

            if 'running_reward' not in globals():
                print("running_reward not in globals")
                running_reward = ep_rs_sum
            else:
                running_reward = running_reward * GAMMA + ep_rs_sum * (1-GAMMA)
            if running_reward > DISPLAY_REWARD_THRESHOLD: RENDER = True     # rendering

            print("episode:", i_episode, "  reward:", int(ep_rs_sum)," runing:",int(running_reward))

            rewards[0].append(int(ep_rs_sum))
            rewards[1].append(int(running_reward))

            vt = RL.learn()

            if i_episode == 0:
                plt.plot(vt)    # plot the episode vt
                plt.xlabel('episode steps')
                plt.ylabel('normalized state-action value')
                plt.show()

            if i_episode%1000 == 0:
                print("saving model...")
                RL.save_model('./model/step'+str(i_episode))
            break

        observation = observation_
ofile = open('sw_0.001_0.99','wb')
pickle.dump(rewards,ofile)
ofile.close()
plt.plot([i for i in range(1,len(rewards[0])+1)],rewards[0],'r-')
plt.plot([i for i in range(1,len(rewards[0])+1)],rewards[1],'b-')
plt.xlabel('episodes')
plt.ylabel('reward')
plt.show()
