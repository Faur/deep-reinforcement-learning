### Standard interface
# env_name = 'CartPole-v0'
# env_max_step = 200
env_name = 'CartPole-v1'
env_max_step = 500

num_state = [4]
num_action = 2

eps = 1e-10 # Small offset to prevent NAN etc.

### Model specific
model_type = 'dense'
# layers = [16]
layers = [64]
# layers = [128]
gamma = 0.99
# gamma = 0.95



## Policy Gradient / Actor Critic
loss_v_coef = 0.5
loss_entropy_coef = 0.01


## DQN
# replay_buffer_size = int(1e6) #Human paper
replay_buffer_size = int(2e5)
replay_min_size = 50000 # Human paper
update_freq = 4 # Human paper
batch_size = 64 # CC paper
# batch_size = 32

eps_start = 1
eps_end = 0.1
# eps_anneal_period = int(1e6) # Human paper
eps_anneal_period = int(5e5)

tau = 0.001 #Rate to update target network toward primary network. CC paper: tau = 0.001


