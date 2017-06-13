import os
import sys
sys.path.append(os.path.join('.', '..')) 
import tensorflow as tf

from tensorflow.contrib.keras.api.keras.layers import Dense, Input, Conv2D, Flatten
from tensorflow.contrib.keras.api.keras.models import Model

def DQN(input_ph, a_size):
    # TODO: Check if the None from the batch size should be excluded in input_ph!
    with tf.name_scope('DQN'):
        input_layer = Input(tensor=input_ph)
        x = Conv2D(32, (8,8), (4,4), activation='relu')(input_layer)
        x = Conv2D(64, (4,4), (2,2), activation='relu')(x)
        x = Conv2D(64, (3,3), (1,1), activation='relu')(x)
        x = Flatten()(x)
        x = Dense(512, activation='relu', name='Dense1')(x)
        x = Dense(a_size, activation=None, name='qValue')(x)

        model = Model(inputs=input_layer, outputs=x)

        return model


def simple_dense(input_ph, a_size, layers, activation='relu', last_activation=None):
    with tf.name_scope('simpleDense'):
        input_layer = Input(tensor=input_ph)
        x = input_layer
        for i in range(len(layers)):
            # print('i', i)
            x = Dense(layers[i], activation=activation, name='Dense'+str(i))(x)

        x = Dense(a_size, activation=last_activation, name='output')(x)

        model = Model(inputs=input_layer, outputs=x)

        return model
