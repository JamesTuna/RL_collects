# RL_collects
## experiments of reinforcement learning algorithms

### Deep Q-network
- Simple version implemented, try it on cartPole game   
- Further improvement to try:   
  - Dueling structure of function approximate network based on [this paper](https://arxiv.org/pdf/1511.06581.pdf)
  - Avoid overestimation by using double Q-learning, based on [this paper](https://arxiv.org/pdf/1509.06461.pdf)   
  - Using prioritzed experience replay.  
  
### vallinaPG
- Implemented for discrete action space
- Basic policy gradient with monte-carlo return
- To try 2 forms of Policy gradient theorem, modify ```Actor._discount_and_norm_rewards()```in policy_gradient.py
  - Current version: Rewards after action (equivalent one, less variance)
  - Uncomment: total rewards of episode (basic version of formula)
- Use moving average of episdoes' return as basline 
- Run ```run_CartPole.py``` to play cartPole balancing game in OpenAI gym.

### PPO
- use clipped surrogate objective
- baseline given by state value approximated by critor
- based on [this paper](https://arxiv.org/abs/1707.06347) 
- Use ```MAX_STEPS``` to control the length of episode you want to play (2000 by default)
- Note that there exists an upper bound of total rewards due to ```MAX_STEPS```
- Next thing to be done: extend this algorithm to
  - Multi-process/thread (multi-actor, single critor; less correlation between experiences)
  - Continuous action selection model   
### Test on [OpenAI gym](https://gym.openai.com/) games.   
![landerdemo](https://user-images.githubusercontent.com/25345821/47616297-c151b400-daf5-11e8-8cc0-2f1840df321e.gif)   
![Training Curve2](https://github.com/JamesTuna/RL_collects/blob/master/experiment_results/lander_smth-1.png)   
Trained agent on LunarLander task. 5000 episodes of training, see hyperparameters in ```experiment_results/RL_set```.   

![mtcar](https://user-images.githubusercontent.com/25345821/47627790-09f88400-db6d-11e8-847d-707037c8e59a.gif)   
![Training Curve2](https://github.com/JamesTuna/RL_collects/blob/master/experiment_results/mtCar_set1.png)   
Trained agent on mountain car game, ```MAX_STEPS``` set to 5000, hence the longest time step each episode can go on is 5000. See detailed hyperparameters setting in ```experiment_results/RL_set```)    

![cartpole](https://user-images.githubusercontent.com/25345821/47627814-31e7e780-db6d-11e8-89fc-72843bf1d76c.gif)     
![Training Curve](https://github.com/JamesTuna/RL_collects/blob/master/experiment_results/cartPoleSet8-1.png)
Trained agent on cartPole game, ```MAX_STEPS``` set to 5000, hence the maximum total reward can be obtained is 5000. See detailed hyperparameters setting in ```experiment_results/RL_set```)     




