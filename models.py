import os
import sys
sys.path.append(os.path.join('.', '..')) 
import tensorflow as tf

from tensorflow.contrib.keras.api.keras.layers import Dense, Input, Conv2D, Flatten
from tensorflow.contrib.keras.api.keras.models import Model

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