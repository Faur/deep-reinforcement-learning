from __future__ import absolute_import, division, print_function, unicode_literals

import gym
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from tensorflow.contrib.keras.api.keras.layers import Dense, Input
from tensorflow.contrib.keras.api.keras.models import Model
from matplotlib.font_manager import FontProperties

# Custom libraries
import utils
import networks
import Logger

class PolicyGradient:
    def __init__(self, config, logdir, learning_rate, max_train_frame=1e6, render=False):
        self.should_stop = False
        self.frame = 0
        self.episode = 0
        self.train_interval = 1 # episodes

        self.config = config
        self.logdir = logdir
        self.learning_rate = learning_rate
        self.max_train_frame = max_train_frame

        self.render = render
        
        self.obsPH = tf.placeholder(tf.float32, shape=[None]+[self.config.num_state], name='obsPlaceholder')
        self.actionPH = tf.placeholder(tf.int32, shape=[None], name='actionPlaceholder')
        self.learningRatePH = tf.placeholder(tf.float32, shape=[], name='learningratePlaceholder')
        self.advantagePH = tf.placeholder(tf.float32, shape=[None], name='advantagePlaceholder')

    def build(self):
        self.env = gym.make(self.config.env_name)
        self.annealer = utils.Annealer(self.learning_rate, 0, self.max_train_frame)
        self.model = self._build_model()
        self.graph = self._build_graph(self.learningRatePH)

        self.sess = tf.Session()
        self.summary_writer = tf.summary.FileWriter(self.logdir, self.sess.graph)
        self.logger = Logger.Logger(self.logdir)
        self.logger.writer = self.summary_writer
        self.saver = tf.train.Saver(max_to_keep=5)
        self.sess.run(tf.global_variables_initializer())

    def _build_model(self):
        input_layer = Input(tensor=self.obsPH)
        model_layers = networks.build_dense(input_layer, self.config.layers, name_stem='dense_')
        model = Model(inputs=input_layer, outputs=model_layers)
        return model

    def _build_graph(self, learning_rate):
        class Graph: pass
        graph = Graph

        action_hot = tf.one_hot(self.actionPH, self.config.num_action)
        with tf.variable_scope('actor'):
            logits = Dense(self.config.num_action, activation='linear')(self.model.output)
            graph.action_probs = tf.nn.softmax(logits)
            graph.action_prob = tf.reduce_sum(graph.action_probs * action_hot,
                                    axis=1, keep_dims=True)

        with tf.variable_scope('training'):
            graph.loss_policy = tf.nn.softmax_cross_entropy_with_logits(
                labels=action_hot, logits=logits)
            graph.loss_policy = tf.reduce_mean(graph.loss_policy * self.advantagePH)
            
            graph.loss_entropy = self.config.loss_entropy_coef * tf.reduce_mean(
                graph.action_probs * tf.log(graph.action_probs + self.config.eps))

            graph.loss_total = graph.loss_policy + graph.loss_entropy
            
            optimizer = tf.train.RMSPropOptimizer(learning_rate, decay=0.99)
            grads_and_vars = optimizer.compute_gradients(graph.loss_total)
            grads, variables = zip(*grads_and_vars)
#             clipped_gradients, _ = zip(*[(tf.clip_by_value(grad, -1., 1.), var)
#                              for grad, var in grads_and_vars])
            ## WARNING: The output from clip_by_value might be totally wrong!!!
            clipped_gradients, _ = (tf.clip_by_global_norm(grads, 1.))
    
#             grad_check = tf.check_numerics(clipped_gradients, 'check_numerics caught bad numerics')
#             try:
#                 with tf.control_dependencies([grad_check]):
            graph.train_op = optimizer.apply_gradients(zip(clipped_gradients, variables))
#             except InvalidArgument:
#                 print('Bad gradients!')
        
        ## Create summaries
        tf.summary.scalar('training/loss_total', graph.loss_total)
        tf.summary.scalar('training/loss_policy', graph.loss_policy)
        tf.summary.scalar('training/loss_entropy', graph.loss_entropy)

        for g, v in grads_and_vars:
            if g is not None:
                tf.summary.histogram('grad_org/'+v.name[:-2], g)
                tf.summary.histogram('var/'+v.name[:-2], g)
        for g, v in zip(clipped_gradients, variables):
            if g is not None:
                tf.summary.histogram('grad_clip/'+v.name[:-2], g)

        graph.summary = tf.summary.merge_all()
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
        """ Takes a single obs, and returns a single action"""
        [p] = self.sess.run(
                self.graph.action_probs,
                feed_dict={self.obsPH : obs})
        a = np.random.choice(self.config.num_action, p=p)
        return a

    def create_video(self, title, target, num_episodes=1, frame_duration=None, figsize=(8,4)):
        import imageio

        # Setup
        font = FontProperties()
        font.set_family('monospace')
        episode = 0
        # figs = []

        # Initialize
        obs = self.env.reset()
        done = False
        reward_sum = 0
        action_prob = [] # probability of goint right
        action_chosen = []
        t_max = 50
        try:
            with imageio.get_writer(target, duration=frame_duration) as writer:
                while episode < num_episodes:
                    [p] = self.sess.run(
                            self.graph.action_probs,
                            feed_dict={self.obsPH : [obs]})
                    action_prob.append(p[1])
                    a = np.random.choice(self.config.num_action, p=p)
                    action_chosen.append(a)
                    obs, reward, done, _ = self.env.step(a)
                    reward_sum += int(reward)
                    img = self.env.render(mode='rgb_array')

                    if t_max - reward_sum < 10:
                        t_max += 50

                    ## Plotting!
                    fig = plt.figure(figsize=figsize, dpi=240)

                    ax = fig.add_subplot(221)
                    plt.imshow(img)
                    ax.yaxis.set_visible(False)
                    ax.xaxis.set_ticks_position('none')
                    ax.set_xticklabels([])

                    ax = fig.add_subplot(222)
                    plt.axis('off')
                    ax.text(0,0, 'Title:         ' + title
                                +'\nEnvironment: ' + self.config.env_name
                                # +'\nExperiment:  ' + experiment_name # Not available in this scope!
                                +'\nNum. param.  ' + str(utils.num_trainable_param())                        
                                +'\nStep:        ' + str(reward_sum)
                            , fontproperties=font)

                    fig.add_subplot(212)
                    plt.title('Action, 1 = right')
                    plt.plot([0, t_max],[0.5, 0.5],'k',alpha=0.5)
                    plt.plot(action_prob)
                    plt.plot(action_chosen, 'bo', markeredgewidth=0.0, markersize=4, alpha=0.25)
                    plt.xlim([0,t_max])
                    plt.ylim([-0.1, 1.1])

                    plt.tight_layout()
                    # figs.append(fig)

                    fig.canvas.draw()
                    data = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
                    data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
                    writer.append_data(data)
                    plt.close(fig)

                    if done:
                        episode += 1
                        print('Episode {:3}, frames {:4}'.format(episode, reward_sum))
                        obs = self.env.reset()
                        done = False
                        reward_sum = 0
                        action_prob = [] # probability of goint right
                        action_chosen = []
                        t_max = 50
        except KeyboardInterrupt:
            print('KeyboardInterrupt')

        self.env.render(close=True)
    
    def run(self, load_model=False):
        if load_model: self.load_model(self.logdir)
        
        done = False
        obs = self.env.reset()
        experience = [[], [], []]
        rewards = []
        try:
            while self.should_stop is False:
                self.frame += 1
                
                action = self.get_action([obs])
                obs, reward, done, _ = self.env.step(action)
                reward = np.clip(reward, -1, 1)
                if self.render: self.env.render()
                
                # add experience to memory
                rewards.append(reward)
                experience[0].append(obs)
                experience[1].append(action)
                
                if done:
                    self.logger.log_scalar(tag='performance/reward', 
                                           value=sum(rewards),
                                           step=self.frame)

                    self.episode += 1                    
                    if sum(rewards) >= self.config.env_max_step: # if we win make the advantage positive for all
#                         print('sum(rewards) =', sum(rewards))
                        dis_r = 0.01 * np.ones_like(rewards)
                        dis_r = list(dis_r)
                    else: # compute discounted rewards
#                         print('Normal')
                        dis_r = utils.discount_rewards(rewards, self.config.gamma)
                        dis_r = list(dis_r)
                        
#                     print('dis_r', type(dis_r), len(dis_r), type(dis_r[9]))
#                     print(dis_r)
#                     break
                    experience[2] += dis_r

                    rewards = []                
                    done = False
                    obs = self.env.reset()
                
                    if self.episode % self.train_interval == 0: 
                            # currently self.train_interval=1
                        assert len(experience[0]) == len(experience[1]), \
                            "Error: experience lenghts don't allign" + str([len(i) for i in experience])
                        assert len(experience[0]) == len(experience[2]), \
                            "Error: experience lenghts don't allign" + str([len(i) for i in experience])
                        self.logger.log_scalar(tag='training/batch_size', 
                                           value=len(experience[0]),
                                           step=self.frame)
 
                        # stack experience
                        obs_stack = np.vstack(experience[0])
                        action_stack = np.vstack(experience[1])
                        action_stack = np.squeeze(action_stack)
                        reward_stack = np.vstack(experience[2])
                        reward_stack = np.squeeze(reward_stack)
                        # normalize discounted rewardrrrr
                        
                        reward_std = np.std(reward_stack)
                        if np.abs(reward_std) > 1e6:
                            reward_stack = (reward_stack - np.mean(reward_stack))/reward_std
                        else:
                            reward_stack = reward_stack - np.mean(reward_stack)
                        experience = [[], [], []]

#                         print('obs_stack', obs_stack.shape)
#                         print('action_stack', action_stack.shape)
#                         print('reward_stack', reward_stack.shape)
#                         break
                        
                        _, summary = self.sess.run(
                                [self.graph.train_op, self.graph.summary], 
                                feed_dict={self.obsPH : obs_stack,
                                           self.actionPH : action_stack,
                                           self.advantagePH : reward_stack,
                                           self.learningRatePH : self.annealer.linear(self.frame)})

                if self.episode % 25 == 0: # we don't actually want to store this much data!!
                    self.logger.log_scalar('training/learning_rate', self.annealer.linear(self.frame), self.frame)
                    self.summary_writer.add_summary(summary, self.frame)
                
                if self.episode % 500 == 0:
                    print('{:8} model saved:'.format(self.frame), self.logdir)
                    self.save_model(self.logdir + '/model_'+str(self.frame))
                
                if self.frame > self.max_train_frame:
                    print('max_train_frame reached')
                    self.should_stop = True
                
                    
        except KeyboardInterrupt:
            print('KeyboardInterrupt')
        print('Training ended')
        self.env.render(close=True)