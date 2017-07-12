env_name = 'CartPole-v0'
env_max_step = 200

env_name = 'CartPole-v1'
env_max_step = 500


layers = [16]
gamma = 0.99
# gamma = 0.95

num_state = 4
num_action = 2

eps = 1e-10 # Small offset to prevent NAN etc.


## Actor-Critic specific
loss_v_coef = 0.5
loss_entropy_coef = 0.01



