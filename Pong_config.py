### Standard interface
env_name = 'Pong-v0'
env_max_step = 500

num_state = [84, 84, 4]
num_action = 6

eps = 1e-10 # Small offset to prevent NAN etc.

### Model specific
model_type = 'conv'
gamma = 0.99
# gamma = 0.95



## Policy Gradient / Actor Critic
loss_v_coef = 0.5
loss_entropy_coef = 0.01


## DQN
# replay_buffer_size = int(1e6) #Human paper
replay_buffer_size = int(5e5)
replay_min_size = 50000 # Human paper
update_freq = 4 # Human paper
# batch_size = 64 # CC paper for low dim
batch_size = 16 # CC paper for high dim

eps_start = 1
eps_end = 0.1
# eps_anneal_period = int(1e6) # Human paper
eps_anneal_period = int(5e5)

# Rate to update target network toward primary network.
tau = 0.001 # CC paper: tau = 0.001


