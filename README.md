# RL_collects
experiments of reinforcement learning algorithms

vallinaPG
-- Basic policy gradient with monte-carlo return
-- To try 2 forms of Policy gradient theorem, modify Actor._discount_and_norm_rewards() in policy_gradient.py
    --Current version: Rewards after action (equivalent one, less variance)
    --Uncomment: total rewards of episode (basic version of formula)
-- Use moving average of episdoes' return as basline 
