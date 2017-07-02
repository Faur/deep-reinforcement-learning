import os
import sys
sys.path.append(os.path.join('.', '..')) 
import tensorflow as tf
import numpy as np

from tensorflow.contrib.keras.api.keras.layers import Dense, Input, Conv2D, Flatten
from tensorflow.contrib.keras.api.keras.models import Model

import utils

def model_DQN(input_ph):
    # TODO: Check if the None from the batch size should be excluded in input_ph!
    with tf.name_scope('DQN'):
        input_layer = Input(tensor=input_ph)
        x = Conv2D(32, (8,8), (4,4), activation='relu', name='Conv1')(input_layer)
        x = Conv2D(64, (4,4), (2,2), activation='relu', name='Conv2')(x)
        x = Conv2D(64, (3,3), (1,1), activation='relu', name='Conv3')(x)
        x = Flatten()(x)
        x = Dense(512, activation='relu', name='Dense1')(x)

        model = Model(inputs=input_layer, outputs=x)

        return model

def model_dense(input_ph, layers, activation='relu',):
    with tf.name_scope('simpleDense'):
        input_layer = Input(tensor=input_ph)
        x = input_layer
        for i in range(len(layers)):
            name = 'Dense'+str(i)
            x = Dense(layers[i], activation=activation, name=name)(x)

        model = Model(inputs=input_layer, outputs=x)

        return model

class DQN(object):
    """docstring for DQN"""
    def __init__(self, model_type, obsPlaceholder, actionPlaceholder, a_size):
        actions_onehot = tf.one_hot(actionPlaceholder, a_size, dtype=tf.float32, name='actionsOnehot')

        self.a_size = a_size
        ## Create model
        if model_type == 'DQN':
            # Accepts 2d input
            assert len(obsPlaceholder.get_shape()) == 4, "'DQN' only accepst 3d input: shape=[None, w, h, c]"
            self.model = model_DQN(obsPlaceholder)
        elif model_type == 'dense':
            # Accepts 1d input
            assert len(obsPlaceholder.get_shape()) == 2, "'dense' only accepst 1d input: shape=[None, x]"
            layers = [128] # TODO: make this a passable parameter!
            self.model = model_dense(obsPlaceholder, layers)

        ## Create ops
        with tf.name_scope('qvalue'):
            self.Qout = Dense(a_size, activation=None)(self.model.output)
            self.qvalue = tf.reduce_sum(tf.multiply(self.Qout, actions_onehot), axis=1)

        self.action = tf.argmax(self.Qout, axis=1, name='action')

    def create_MSE_train_op(self, targetQPlaceholder, learning_rate):
        with tf.name_scope('loss'):
            self.loss = tf.reduce_mean(tf.square(targetQPlaceholder - self.qvalue)) 

        self.train_op = tf.train.RMSPropOptimizer(learning_rate).minimize(self.loss) 


class Trainer(object):
    def __init__(self, obsBuf, expBuf, annealer, envInter, env, logdir, saver, DQN1, DQN2,
                minimum_experience, update_frequency, track_interval, gamma, max_ep_t=np.inf):
        self.obsBuf = obsBuf
        self.expBuf = expBuf
        self.annealer = annealer
        self.envInter = envInter
        self.env = env
        self.logdir = logdir
        self.saver = saver
        self.DQN1 = DQN1
        self.DQN2 = DQN2
        
        self.minimum_experience = minimum_experience
        self.update_frequency = update_frequency
        self.track_interval = track_interval
        self.gamma = gamma
        self.max_ep_t = max_ep_t
        
        self.frame = 0
        self.should_run = True
        
        def updateTargetGraph(tfVars,tau):
            """ Stuff for updating the target graph """
            # TODO: FIX THIS!!
            # This is a terrible way of doing it, it relies on the order in which the networks were created
            total_vars = len(tfVars)
            op_holder = []
            for idx,var in enumerate(tfVars[0:total_vars//2]):
                op_holder.append(tfVars[idx+total_vars//2].assign((var.value()*tau) + ((1-tau)*tfVars[idx+total_vars//2].value())))
            return op_holder

        tau = 0.0005 #Rate to update target network toward primary network
        trainables = tf.trainable_variables()
        self.targetOps = updateTargetGraph(trainables,tau)

    def updateTarget(self, op_holder,sess):
        """ Stuff for updating the target graph """
        for op in op_holder:
            sess.run(op)
        
    def new_episode(self):
        # Prepare for new episode
        self.obsBuf.reset()
        obs, ep_t, ep_r = self.envInter.reset(self.env)
        self.obsBuf.add(obs)
        net_input = self.obsBuf.get()
        return net_input, ep_t, ep_r

    def train(self, sess, training_summaries, epsilon_test, render_interval, 
              obsPlaceholder, actionPlaceholder, targetQPlaceholder, 
              batch_size, load_model=False):
        ## Load model
        if load_model == True:
            print('Loading Model...')
            ckpt = tf.train.get_checkpoint_state(self.logdir)
            try:
                self.saver.restore(sess, ckpt.model_checkpoint_path)
            except AttributeError:
                if not os.path.exists(self.logdir):
                    os.mkdir(self.logdir)
                print('WARNING: Could not load previous model')
        else:
            if os.path.exists(self.logdir):
                print('Deleting old model')
                shutil.rmtree(self.logdir)
            os.mkdir(self.logdir)


        ## Begin training
        net_input, ep_t, ep_r = self.new_episode()
        try:
            while self.should_run:
                if self.frame == self.minimum_experience:
                    print('\n\n___/// BEGIN TRAINING \\\___')

                epsilon = self.annealer.linear(self.frame - self.minimum_experience)

                if training_summaries['num_ep'] % render_interval == 0 \
                        and self.frame > self.minimum_experience:
                    # TODO: ATM we are still collecting expeirences during this.
                    # this is technically wrong, but who cares
                    epsilon = epsilon_test
                    self.env.render()

                if np.random.rand(1) > epsilon:
                    action = sess.run(self.DQN1.action, feed_dict={obsPlaceholder : [net_input]})
                else:
                    action = np.random.randint(self.DQN1.a_size)
                action = int(action)
                assert 'int' in str(type(action)), 'action must be an int, not a ' + str(type(action))

                next_obs, reward, done, _ = self.envInter.take_action(action, self.env)
                self.obsBuf.add(next_obs)
                net_input_next = self.obsBuf.get()
                ep_t += 1
                ep_r += reward
                reward = np.clip(reward, -1, 1)

                # Save experiences
                experience = {'obs':[net_input], 'action':action, 'reward':reward,
                              'next_obs':[net_input_next], 'done':done}
                self.expBuf.add(experience)
                net_input = net_input_next

                ## Update Weights
                if self.frame % self.update_frequency == 0 \
                        and self.frame > self.minimum_experience:
                    train_batch = self.expBuf.sample(batch_size)

                    # Compute ... TODO: add description
                    [action] = sess.run([self.DQN1.action],
                        feed_dict={obsPlaceholder : train_batch['next_obs']})
                    [actualQs] = sess.run([self.DQN2.Qout],
                        feed_dict={obsPlaceholder : train_batch['next_obs']})

                    actualQ = actualQs[range(batch_size), action] 
                        # The DQN2 Q value of the action chosen by DQN1
                    zero_if_done = train_batch['done']*(-1) + 1 # Used to remove actualQ
                                                                # When at terminal state
                    target = train_batch['reward'] + self.gamma*actualQ*zero_if_done

                    ## Update DQN1
                    DQN1_train_dict = {
                                obsPlaceholder : train_batch['obs'],
                                actionPlaceholder : train_batch['action'],
                                targetQPlaceholder : target
                        }
                    loss, _ = sess.run([self.DQN1.loss, self.DQN1.train_op], feed_dict=DQN1_train_dict)

                    ## Update DQN2
                    self.updateTarget(self.targetOps, sess)          


                if ep_t > self.max_ep_t:
                    done = True

                if done:
                    if training_summaries['num_ep'] % render_interval == 0 \
                            and self.frame > self.minimum_experience:
                        print('{:22} {:9}, {:7}, {:7.1f} <-- Test reward!'.format(
                                utils.time_str(), self.frame, training_summaries['num_ep'], ep_r))

                    training_summaries['num_ep'] += 1
                    training_summaries['ep_rewards'].append(ep_r)
                    training_summaries['ep_length'].append(ep_t)
                    training_summaries['epsilon'].append(epsilon)

                    net_input, ep_t, ep_r = self.new_episode()
        #             break

                ## Track training
                if self.frame % self.track_interval == 0:
                    if (self.frame % (self.track_interval*10)==0) and (self.frame>self.minimum_experience):
                        i = training_summaries['num_ep']
                        model_save_str = self.logdir+'/model-'+str(i)+'.cptk'
                        self.saver.save(sess, model_save_str)
                        print("Saved Model: " + model_save_str)
        #                 break

                    if self.frame % (self.track_interval*25) == 0:
                        print('\n{:22} {:>9}  {:>7}  {:>7}  {:>7}'.format(
                            'time', 'frames', 'epis', 'reward', 'epsilon'))
                    else:
                        print('{:22} {:9}, {:7}, {:7.1f}, {:7.3f}'.format(
                              utils.time_str(),
                              self.frame, 
                              training_summaries['num_ep'],
                              np.mean(training_summaries['ep_rewards'][-100:]),
                              training_summaries['epsilon'][-1],
                             ))

                self.frame += 1
        except KeyboardInterrupt:
            done = True

        self.env.render(close=True)
        print('\nTrainer.train() Terminated')


if __name__ == '__main__':
    a_size = 2
    learning_rate = 1e-3

    obs1D = tf.placeholder(tf.float32, shape=[None, 2], name='obsPlaceholder')
    obs2D = tf.placeholder(tf.float32, shape=[None, 84, 84, 4], name='obsPlaceholder')
    actionPlaceholder = tf.placeholder(tf.int32, shape=[None], name='actionPlaceholder')
    targetQPlaceholder = tf.placeholder(tf.float32, shape=[None], name='targetQPlaceholder')

    print('Test: DQN for CartPole')
    with tf.name_scope('DQN1'):
        DQN1 = DQN('dense', obs1D, actionPlaceholder, a_size)
        DQN1.create_train_op(targetQPlaceholder, learning_rate=learning_rate)
    print('\nSuccessfully created model')

    print('Test: DQN for Breakout')
    with tf.name_scope('DQN2'):
        DQNcnn = DQN('DQN', obs2D, actionPlaceholder, a_size)
        DQNcnn.create_train_op(targetQPlaceholder, learning_rate=learning_rate)
    print('\nSuccessfully created model')


    print('Tests complete')