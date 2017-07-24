from __future__ import absolute_import, division, print_function, unicode_literals

import os
import sys
sys.path.append(os.path.join('.', '..')) 
import tensorflow as tf
import numpy as np

from tensorflow.contrib.keras.api.keras.layers import Dense, Input, Conv2D, Flatten
from tensorflow.contrib.keras.api.keras.models import Model

import utils

def build_DQN(input_layer):
    # with tf.name_scope('DQN'):
    # input_layer = Input(tensor=input_ph)
    x = Conv2D(32, (8,8), (4,4), activation='relu', name='Conv1')(input_layer)
    x = Conv2D(64, (4,4), (2,2), activation='relu', name='Conv2')(x)
    x = Conv2D(64, (3,3), (1,1), activation='relu', name='Conv3')(x)
    x = Flatten()(x)
    x = Dense(512, activation='elu', name='Dense1')(x)
    # model = Model(inputs=input_layer, outputs=x)

    return x

def build_dense(input_layer, layers, activation='relu', name_stem="dense_"):
    """ Create simple feed-forward dense layers
        Args:
        * input_layer: A keras layer, e.g. layers.Input for the first layer
        * layers: list of int, with the number of units desired
    """
    x = input_layer

    for i in range(len(layers)):
        name = name_stem+str(i)
        with tf.variable_scope(name):
            x = Dense(layers[i], activation=activation)(x)

    return x