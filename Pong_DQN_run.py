from __future__ import absolute_import, division, print_function, unicode_literals

## Standard libraries
import os
import shutil
import time

## 3rd party
import numpy as np
import matplotlib.pyplot as plt
# %matplotlib inline
import tensorflow as tf
import gym

## Custom
# import modelDQN
import networks
import utils
import Pong_config as config
import modelDQN




## Script behavior
experiment_name = None
experiment_name = 'conv'
IS_DEBUGGING = True # Make progam run faster
IS_DEBUGGING = False

## Training
# learning_rate = 5e-4
learning_rate = 0.00025 # Human paper
# max_train_frame = 2e6
max_train_frame = 2e6
max_train_frame = 5e7
max_episode_frame = 5000



if IS_DEBUGGING:
    config.eps_anneal_period = 1e5
    config.replay_buffer_size = int(1e3)



## Derived settings
prepro = utils.Preprocessor_2d(config.num_state, gray=True)
env = utils.EnvironmentInterface(config, prepro, action_repeats=4, obs_buffer_size=4)

for i in range(1):
    print('i', i)
#     run_name = experiment_name or utils.time_str()
    run_name = experiment_name + utils.time_str()
    logdir = './logdir/'+config.env_name+'/DQN/' + run_name
    print('logdir\t\t', logdir)
    if IS_DEBUGGING: logdir += '_debug'

    ## Setup
    tf.reset_default_graph()
    random_seed = int(time.time())
    tf.set_random_seed(random_seed)
    np.random.seed(random_seed)
    print('random seed\t', random_seed)

    print('Creating new model')
    agent = modelDQN.DQN(config, env, logdir, learning_rate, max_train_frame=max_train_frame, max_episode_frame=max_episode_frame)
    agent.build()
    agent.run(load_model=False)

#     agent.create_video('DQN', './tmp/CP_DQN_' + run_name + '.gif')
    
print('done')