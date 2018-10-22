# RL_collects
## experiments of reinforcement learning algorithms

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
- Note that there is an upper bound rewards due to ```MAX_STEPS```
- Next thing to be done: extend this algorithm to
  - 1. Multi-process/thread
  - 2. Continuous action selection model
