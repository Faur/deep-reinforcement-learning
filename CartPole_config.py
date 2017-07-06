env_name = 'CartPole-v1'
layers = [16]
gamma = 0.99

num_state = 4
num_action = 2

eps = 1e-10 # Small offset to prevent NAN etc.



## Actor-Critic specific
loss_v_coef = 0.5
loss_entropy_coef = 0.01



