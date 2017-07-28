from __future__ import absolute_import, division, print_function, unicode_literals

## Standard libraries
import time

## 3rd party libraries
import tensorflow as tf
import numpy as np
# %matplotlib inline
# import matplotlib.pyplot0as plt
# import numpy as np
import gym

## Custom libraries
import modelPolicyGradient
import utils
import Pong_config as config


## Settings and parameters
experiment_name = None
# experiment_name = '2017-07-13-(05-16-25)' # Complete shit
# experiment_name = '2017-07-13-(05-59-03)' # Complete shit
# experiment_name = '2017-07-12-(17-09-11)' # Almost shit


## Training settings
learning_rate = 1e-3
max_train_frame = 5e7



## Derived settings
run_name = experiment_name or utils.time_str()
logdir = './logdir/'+config.env_name+'/PG/' + run_name
print('logdir\t\t', logdir)




num_run = 1
# env = gym.make(config.env_name)

prepro = utils.Preprocessor_2d(config.num_state, gray=True)
env = utils.EnvironmentInterface(config, prepro, action_repeats=4, obs_buffer_size=4)

for i in range(num_run):
    run_name = experiment_name or utils.time_str()
    logdir = './logdir/'+config.env_name+'/PG/' + run_name
    print('logdir\t\t', logdir)

    ## Setup
    tf.reset_default_graph()
    random_seed = int(time.time())
    tf.set_random_seed(random_seed)
    np.random.seed(random_seed)
    print('random seed\t', random_seed)

    ## Build model
    agent = modelPolicyGradient.PolicyGradient(config=config, env=env,
                  learning_rate=learning_rate, logdir=logdir, 
                  max_train_frame=max_train_frame,  render=False)
    agent.build()
    agent.run(load_model=True)
    print('done')
    print('\n\n')