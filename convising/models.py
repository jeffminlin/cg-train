import numpy as np

from tensorflow.keras.layers import Dense, Lambda, Input, Conv2D, Activation
from tensorflow.keras.layers import Add, Multiply, Subtract, Dot, Reshape
from tensorflow.keras.models import Model
import tensorflow as tf

import convising.layers as lys


def log_cosh(x):

    return tf.log((tf.exp(x) + tf.exp(-x))/2.0)


def deep_conv_e(config, conv_activ, activation_fcn, kninit):
    """RBM with additional dense layers

    Args:
        config (Config): Configuration as defined in convising.train
        conv_activ: Activation function to use on the convolutional layer
        activation_fcn: Activation function to use on the dense layers
        kninit: Initializer for the kernel weights matrix

    Returns:
        Model: Returns E

    """

    M_in = Input(shape=(None, None))
    M_pad = lys.PeriodicPad2D(name='pad', pad_size=config.w_size-1)(M_in)
    if config.conv_activ == 'log_cosh':
        M_conv = lys.Conv2DNFSym(config.alpha, [config.w_size, config.w_size], strides=(1,1), activation=log_cosh, padding='valid', use_bias=False, nfsym=config.nfsym)(M_pad)
    else:
        M_conv = lys.Conv2DNFSym(config.alpha, [config.w_size, config.w_size], strides=(1,1), activation=config.conv_activ, padding='valid', use_bias=False, nfsym=config.nfsym)(M_pad)
    if config.nfsym == 'z2' or config.nfsym == 'all':
        M_conv = Lambda(lambda x: tf.expand_dims(x[0]) * x[1])([M_in, M_conv])
    if len(config.dense_nodes) > 1:
        M_fc = Dense(config.dense_nodes[0], activation=activation_fcn, use_bias=True, kernel_initializer=kninit)(M_conv)
        for nodenum in config.dense_nodes[1:]:
            M_fc = Dense(nodenum, activation=activation_fcn, use_bias=False, kernel_initializer=kninit)(M_fc)
    else:
        M_fc = Dense(config.dense_nodes[0], activation=activation_fcn, use_bias=True, kernel_initializer=kninit)(M_conv)
    M_sum = Lambda(lambda x: tf.sum(x, axis=[1,2]), name='sum_over_spins')(M_fc)
    M_lincomb = Dense(1, activation='linear', use_bias=False, kernel_initializer=kninit)(M_sum)
    model_energy = Model(inputs=M_in, outputs=M_lincomb)

    return model_energy


def linear_basis(config, kninit):

    M_in = Input(shape=(None, None))
    M_basis = lys.LinearBasis()(M_in)
    M_lincomb = Dense(1, activation='linear', use_bias=False, kernel_initializer=kninit)(M_basis)
    model_energy = Model(inputs=M_in, outputs=M_lincomb)

    return model_energy


def model_e_diff(config, model_energy):

    M_in_concat = [Input(shape=(None,None)),Input(shape=(None,None))]
    M_energy_diff = Subtract(name='energy_diff')([model_energy(M_in_concat[1]), model_energy(M_in_concat[0])])
    M_exp = Lambda(lambda x: tf.exp(-config.beta*x), name='exp')(M_energy_diff)
    model = Model(inputs=M_in_concat, outputs=M_exp)

    return model
