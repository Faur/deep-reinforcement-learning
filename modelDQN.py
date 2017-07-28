from __future__ import absolute_import, division, print_function, unicode_literals

## Standard libraries

## 3rd party libraries
import tensorflow as tf
import numpy as np
import gym 
from tensorflow.contrib.keras.api.keras.layers import Dense, Input, Conv2D, Flatten
from tensorflow.contrib.keras.api.keras.models import Model
from matplotlib.font_manager import FontProperties
import matplotlib.pyplot as plt

## Custom libraries
import utils
import networks
import Logger

class DQN(object):
    """docstring for DQN""" 
    def __init__(self, config, env, logdir, learning_rate, max_train_frame=5e7, max_episode_frame=int(1e4),
                 render=False):
        self.should_stop = False
        self.frame = -1 # Also create summaries on first run
        self.episode = -1
        self.replay_buffer_size = config.replay_buffer_size

        self.config = config
        self.env = env
        self.logdir = logdir
        self.learning_rate = learning_rate
        self.max_train_frame = max_train_frame
        self.max_episode_frame = max_episode_frame
        
        self.render = render
        
        self.obsPH = tf.placeholder(tf.float32, shape=[None]+self.config.num_state, name='obsPlaceholder')
        self.actionPH = tf.placeholder(tf.int32, shape=[None], name='actionPlaceholder')
        self.learningRatePH = tf.placeholder(tf.float32, shape=[], name='learningratePlaceholder')
        self.targetQPH = tf.placeholder(tf.float32, shape=[None], name='targetQPlaceholder')
        self.tauPH = tf.placeholder(tf.float32, shape=[], name='tauPlaceholder')

    def build(self):
        ## Build helpers
        self.lr_annealer = utils.Annealer(self.learning_rate, self.learning_rate/10, self.max_train_frame)
        self.eps_annealer = utils.Annealer(self.config.eps_start,
                                           self.config.eps_end,
                                           self.config.eps_anneal_period)
        self.expBuf = utils.Experience_buffer(self.replay_buffer_size)
        
        ## Build model and main graph
        if self.config.model_type == 'dense':
            self.DQN_std, self.DQN_tgt = self._build_dense_model()
        elif self.config.model_type == 'conv':
            self.DQN_std, self.DQN_tgt = self._build_conv_model()
        else:
            print("ERROR:", self.config.model_type, "is an unrecognized model type.")
        self.graph = self._build_graph(self.DQN_std, self.DQN_tgt)
        
        ## Setup and finalize
        self.sess = tf.Session()
        self.summary_writer = tf.summary.FileWriter(self.logdir, self.sess.graph)
        self.logger = Logger.Logger(self.logdir)
        self.logger.writer = self.summary_writer
        self.saver = tf.train.Saver(max_to_keep=5)
        self.sess.run(tf.global_variables_initializer())
        self.sess.run(self.graph.update_tgt, {self.tauPH : 1}) # set target graph = std graph
        
    def _build_dense_model(self):
        input_layer = Input(tensor=self.obsPH)
        
        with tf.variable_scope('DQN_std'): # std graph must be constructed before tgt!
            model_layers = networks.build_dense(input_layer, self.config.layers)
            DQN_std = Model(inputs=input_layer, outputs=model_layers)
        
        with tf.variable_scope('DQN_tgt'):
            model_layers = networks.build_dense(input_layer, self.config.layers)
            DQN_tgt = Model(inputs=input_layer, outputs=model_layers)
        
        return DQN_std, DQN_tgt

    def _build_conv_model(self):
        input_layer = Input(tensor=self.obsPH)
        
        with tf.variable_scope('DQN_std'): # std graph must be constructed before tgt!
            model_layers = networks.build_conv(input_layer)
            DQN_std = Model(inputs=input_layer, outputs=model_layers)
        
        with tf.variable_scope('DQN_tgt'):
            model_layers = networks.build_conv(input_layer)
            DQN_tgt = Model(inputs=input_layer, outputs=model_layers)
        
        return DQN_std, DQN_tgt

    
    def _build_graph(self, model_std, model_tgt):
        ## TODO: Ideally this should be broken into two part
            # _build_graph_forwards  - builds the action and qValue, that is called twice
            # _build_graph_backwards - builds the training ops, called once
        class Graph: pass
        graph = Graph
                
        with tf.variable_scope('DQN_std'):
            ## Create standard graph forward pass
            actions_hot = tf.one_hot(self.actionPH, self.config.num_action, 
                                        dtype=tf.float32, name='actionsOnehot')            
            with tf.variable_scope('qValue'):
                graph.qValues = Dense(self.config.num_action, activation=None)\
                                    (model_std.output)
                graph.qValue = tf.reduce_sum(tf.multiply(graph.qValues, actions_hot), axis=1)
            graph.action = tf.argmax(graph.qValues, axis=1, name='argmaxAction')
            
            ## Create standard graph backwards pass
            with tf.variable_scope('training'):
                with tf.variable_scope('loss'):
                    with tf.variable_scope('q'):
                        graph.loss_q = tf.reduce_mean(tf.square(self.targetQPH - graph.qValue))
                    graph.loss_total = graph.loss_q
                
                optimizer = tf.train.RMSPropOptimizer(self.learningRatePH, decay=0.99)
#                 grads, variables = zip(* optimizer.compute_gradients(graph.loss_total))
#                 clipped_grads, _ = (tf.clip_by_global_norm(grads, 0.1))
#                 graph.train_op = optimizer.apply_gradients(zip(clipped_grads, variables))


#                 grads_and_vars = optimizer.compute_gradients(graph.loss_total)
                grads, variables = zip(* optimizer.compute_gradients(graph.loss_total))
                grads = [grad if grad is not None else tf.zeros_like(var) 
                                      for grad, var in zip(grads, variables)]
                    # Make 'none' grads into zeros
                clipped_grads_and_vars = [(tf.clip_by_value(grad, -1., 1.), var) for grad, var in zip(grads, variables)]
                graph.train_op = optimizer.apply_gradients(clipped_grads_and_vars)
                
        with tf.variable_scope('DQN_tgt'):
            ## Create target graph forward pass
            with tf.variable_scope('qValue'):
                graph.qValues_tgt = Dense(self.config.num_action, activation=None)\
                                    (model_tgt.output)
            graph.action_tgt = tf.argmax(graph.qValues_tgt, axis=1, name='argmaxAction')

            ## Create target update op
            with tf.variable_scope('update'):
                std_vars, tgt_vars = [], []
                for var in tf.trainable_variables():
                    if var.name.startswith('DQN_std'):
                        std_vars.append(var)
                    elif var.name.startswith('DQN_tgt'):
                        tgt_vars.append(var)

                graph.update_tgt = []
                for std_var, tgt_var in zip(std_vars, tgt_vars):
                    op = tgt_var.assign(
                        (1-self.tauPH)*tgt_var.value() + self.tauPH*std_var.value())
                    graph.update_tgt.append(op)

        ## Create summaries
        tf.summary.scalar('training/loss_total', graph.loss_total)

        for g, v in zip(grads, variables):
            if (g is not None) and (v is not None):
                tf.summary.histogram('grad_org/'+v.name[:-2], g)
#                 tf.summary.histogram('var/'+v.name[:-2], g)
        for g, v in clipped_grads_and_vars:
            if (g is not None) and (v is not None):
                tf.summary.histogram('grad_clip/'+v.name[:-2], g)

        graph.summary_op = tf.summary.merge_all()
        return graph
    
    def load_model(self, path):
        try:                
            ckpt = tf.train.get_checkpoint_state(path)
            self.saver.restore(self.sess, ckpt.model_checkpoint_path)
        except:
            print("Could not find model to load.")

    def save_model(self, path):
        self.saver.save(self.sess, path)

    def stop(self):
        self.should_stop = True
    
    def get_action(self, obs):
        eps = self.eps_annealer.linear(self.frame)
        if np.random.rand(1) > eps:
            action = self.sess.run(self.graph.action, feed_dict={self.obsPH : obs})
        else:
            action = np.random.randint(self.config.num_action)
        return int(action)
    
    def random_fill_experience_buffer(self, min_size=None):
        """ Fill the experience buffer with random actions"""

        obs = self.env.reset()
        if min_size is None:
            min_size = self.expBuf.buffer_capacity

        print('Filling experience buffer:', min(min_size, self.expBuf.buffer_capacity))        
        while self.expBuf.buffer_size() <= min_size:
            if self.expBuf.is_full():
                break
            
            action = np.random.randint(self.config.num_action)
            obs_next, reward, done, _ = self.env.step(action)
            reward = np.clip(reward, -1, 1)
            done = int(done)
            experience = {'obs':[obs], 'action':action, 'reward':reward,
                      'next_obs':[obs_next], 'done':done}
            self.expBuf.add(experience)
                        
            if done:
                obs = self.env.reset()
            
            
    
    def run(self, load_model=False):
        if load_model: self.load_model(self.logdir)

        try:
            self.random_fill_experience_buffer(self.config.replay_min_size)
            
            print('Begin train loop')
            done = False
            obs = self.env.reset()
            ep_t = 0 # episode time step
            ep_r = 0 # episode total reward
            ep_r_clip = 0 # episode total reward
            while self.should_stop is False:
                self.frame += 1
                action = self.get_action([obs])
#                 print('action', action, type(action))
                
                obs_next, reward, done, _ = self.env.step(action)
                ep_t += 1
                ep_r += reward
                reward = np.clip(reward, -1, 1)
                ep_r_clip += reward
                done = int(done)
                experience = {'obs':[obs], 'action':action, 'reward':reward,
                              'next_obs':[obs_next], 'done':done}
                if ep_r != 500: # TODO: Remove this???
                    self.expBuf.add(experience)
                else:
                    # IGNORE TERMINAL DUE TO SUCCESS!
                    print('EXP ignored because ep_r == 500!!')
                obs = obs_next


                if self.frame % self.config.update_freq == 0:
                    ## Update parameters
                    train_batch = self.expBuf.sample(self.config.batch_size)
                    
                    [action, qValues_tgt] = self.sess.run([self.graph.action, self.graph.qValues_tgt],
                        feed_dict={self.obsPH : train_batch['next_obs']})
                    
                    qValue_tgt = qValues_tgt[range(self.config.batch_size), action] 
#                         # The DQN2 Q value of the action chosen by DQN1
                    terminal_mask = 1 - train_batch['done'] # Remove qValue_tgt, when at terminal state
    
                    target = train_batch['reward'] + self.config.gamma*qValue_tgt*terminal_mask

                    ## Update DQN1
                    DQN1_train_dict = {
                                self.obsPH : train_batch['obs'],
                                self.actionPH : train_batch['action'],
                                self.targetQPH : target,
                                self.learningRatePH : self.lr_annealer.linear(self.frame),
                                self.tauPH : self.config.tau,
                        }
                    summary, _, _ = self.sess.run([self.graph.summary_op, self.graph.train_op, self.graph.update_tgt],
                                      feed_dict=DQN1_train_dict)

                if self.frame % int(1e3) == 0:
		    print('logging\t', self.frame, '\t', self.episode)
                    ## Tensorboard logging
                    self.summary_writer.add_summary(summary, self.frame)
                    self.logger.log_scalar('training/learning_rate', self.lr_annealer.linear(self.frame), self.frame)
                    self.logger.log_scalar('training/epsilon', self.eps_annealer.linear(self.frame), self.frame)
                    
        
                if self.frame % int(5e5) == 0\
                        and self.frame > 0:
                    ## save model
                    self.save_model(self.logdir + '/model_'+str(self.frame))
                    print('{:10} model saved:'.format(self.frame), self.logdir)

                
                if ep_t > self.max_episode_frame:
                    done = True
                
                if done:
#                     print(ep_t, ep_r, ep_r_clip)
                    self.episode += 1
                    self.logger.log_scalar('performance/episode_len', ep_t, self.frame)
                    self.logger.log_scalar('performance/reward',      ep_r, self.frame)
                    self.logger.log_scalar('performance/reward_clip', ep_r_clip, self.frame)
                    self.logger.log_scalar('performance/episodes',    self.episode, self.frame)
                    done = False
                    obs = self.env.reset()
                    ep_t = 0 
                    ep_r = 0 
                    ep_r_clip = 0 
#                     break

                if self.frame > self.max_train_frame:
                    print('max_train_frame reached')
                    self.should_stop = True
                
        except KeyboardInterrupt:
            print('KeyboardInterrupt')
        print('Training ended')
        self.env.render(close=True)
    
    
    def create_video(self, title, target_dir, num_episodes=1, frame_duration=None, figsize=(8,4)):
        import imageio

        ## Setup
        font = FontProperties()
        font.set_family('monospace')
        episode = 0

        ## Initialize
        obs = self.env.reset()
        done = False
        reward_sum = 0
        Qs_collection = np.zeros((1, self.config.num_action))
        t_max = 50
        ep_t = 0
        actions = []
        try:
            with imageio.get_writer(target_dir, duration=frame_duration) as writer:
                while episode < num_episodes:
                    ep_t += 1
                    [Qs] = self.sess.run(
                            self.graph.qValues,
                            feed_dict={self.obsPH : [obs]})
                    action = np.argmax(Qs)
                    actions.append(action)
                    Qs = np.expand_dims(Qs, axis=0)
                    Qs_collection = np.concatenate((Qs_collection, Qs), axis=0)
                    
                    obs, reward, done, _ = self.env.step(action)
                    reward_sum += reward
                    img = self.env.render(mode='rgb_array')


                    if t_max - ep_t < 10:
                        t_max += 50
                    
                    ## Plotting!
                    fig = plt.figure(figsize=figsize, dpi=240)

                    ## Image
                    ax = fig.add_subplot(221)
                    plt.imshow(img)
                    ax.yaxis.set_visible(False)
                    ax.xaxis.set_ticks_position('none')
                    ax.set_xticklabels([])

                    ## Text
                    ax = fig.add_subplot(222)
                    plt.axis('off')
                    ax.text(0,0,   'Title:       ' + title
                                +'\nEnvironment: ' + self.config.env_name
                                # +'\nExperiment:  ' + experiment_name # Not available in this scope!
                                +'\nNum. param.  ' + str(utils.num_trainable_param())                        
                                +'\nStep:        ' + str(ep_t)
                                +'\nReward:      ' + str(reward_sum)                            
                                +'\nAction:      ' + str(action)                            
                            , fontproperties=font)

                    ## Q-value
                    fig.add_subplot(223)
                    plt.title('Q values')
#                     plt.plot([0, t_max],[0.5, 0.5],'k',alpha=0.5)
                    for i in range(Qs.shape[1]):
                        plt.plot(Qs_collection[1:,i],)
#                     plt.plot(action_chosen, 'bo', markeredgewidth=0.0, markersize=4, alpha=0.25)
                    plt.xlim([0,t_max])
                    
                    ## Action
                    fig.add_subplot(224)
                    plt.title('Action')
                    plt.plot(actions)
                    plt.ylim([-0.1, self.config.num_action-0.9])
                    plt.xlim([0,t_max])

                    plt.tight_layout()

                    fig.canvas.draw()
                    data = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
                    data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
                    writer.append_data(data)
#                     plt.close(fig)
                    from IPython.display import clear_output
                    clear_output(wait=True)
                    plt.show()

                    if done:
                        episode += 1
                        Qs_collection = np.zeros((1,2))
                        print('Episode {:3}, frames {:4}'.format(episode, reward_sum))
                        obs = self.env.reset()
                        done = False
                        reward_sum = 0
                        t_max = 50
                        ep_t = 0
                        actions = []
        except KeyboardInterrupt:
            print('KeyboardInterrupt')
        self.env.render(close=True)
