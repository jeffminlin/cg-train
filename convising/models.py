import numpy as np
import tensorflow as tf

import convising.layers as lys


def log_cosh(x):

    return tf.math.log((tf.math.exp(x) + tf.math.exp(-x)) / 2.0)


def get_activation(activation_name):

    if activation_name == "log_cosh":
        return log_cosh
    else:
        return activation_name


def deep_conv_e(conv_activation, nfilters, kernel_size, dense_nodes, dense_activation):
    """RBM with additional dense layers

    Args:
        config (Config): Configuration as defined in convising.train
        conv_activ: Activation function to use on the convolutional layer
        activation_fcn: Activation function to use on the dense layers
        kninit: Initializer for the kernel weights matrix

    Returns:
        Model: Returns E

    """

    M_in = tf.keras.layers.Input(shape=(None, None))
    M_pad = lys.PeriodicPad2D(name="periodic_pad", pad_size=kernel_size - 1)(M_in)
    M_conv = tf.keras.layers.Conv2D(
        nfilters,
        [kernel_size, kernel_size],
        strides=(1, 1),
        activation=get_activation(conv_activation),
        padding="valid",
        use_bias=False,
        name="convolution",
    )(M_pad)
    if len(dense_nodes) > 1:
        M_fc = tf.keras.layers.Dense(
            dense_nodes[0], activation=dense_activation, use_bias=True, name="dense_0"
        )(M_conv)
        for idx, nodenum in enumerate(dense_nodes[1:]):
            M_fc = tf.keras.layers.Dense(
                nodenum,
                activation=dense_activation,
                use_bias=False,
                name="dense_" + str(idx + 1),
            )(M_fc)
    else:
        M_fc = tf.keras.layers.Dense(
            dense_nodes[0], activation=dense_activation, use_bias=True, name="dense"
        )(M_conv)
    M_sum = tf.keras.layers.Lambda(
        lambda x: tf.math.reduce_sum(x, axis=[1, 2]), name="sum_over_spins"
    )(M_fc)
    M_lincomb = tf.keras.layers.Dense(
        1, activation="linear", use_bias=False, name="combine_basis"
    )(M_sum)
    model_energy = tf.keras.Model(inputs=M_in, outputs=M_lincomb, name="deep_energy")

    return model_energy


def conv_multiply(nfilters, kernel_size, dense_nodes, dense_activation):

    M_in = tf.keras.layers.Input(shape=(None, None))
    M_pad = lys.PeriodicPad2D(name="periodic_pad", pad_size=kernel_size - 1)(M_in)
    M_conv1 = tf.keras.layers.Conv2D(
        nfilters,
        [kernel_size, kernel_size],
        strides=(1, 1),
        activation="linear",
        padding="valid",
        use_bias=False,
        name="convolution_1",
    )(M_pad)
    M_conv2 = tf.keras.layers.Conv2D(
        nfilters,
        [kernel_size, kernel_size],
        strides=(1, 1),
        activation="linear",
        padding="valid",
        use_bias=False,
        name="convolution",
    )(M_pad)
    M_mult = tf.keras.layers.Multiply()([M_conv1, M_conv2])
    M_fc = M_mult
    for idx, nodenum in enumerate(dense_nodes):
        M_fc = tf.keras.layers.Dense(
            nodenum,
            activation=dense_activation,
            use_bias=False,
            name="dense_" + str(idx),
        )(M_fc)
    M_sum = tf.keras.layers.Lambda(
        lambda x: tf.math.reduce_sum(x, axis=[1, 2]), name="sum_over_spins"
    )(M_fc)
    M_lincomb = tf.keras.layers.Dense(
        1, activation="linear", use_bias=False, name="combine_basis"
    )(M_sum)
    model_energy = tf.keras.Model(inputs=M_in, outputs=M_lincomb, name="conv_multiply")

    return model_energy
    


def linear_basis():

    M_in = tf.keras.layers.Input(shape=(None, None))
    M_basis = lys.LinearBasis(name="sum_over_spins")(M_in)
    M_lincomb = tf.keras.layers.Dense(
        1, activation="linear", use_bias=False, name="combine_basis"
    )(M_basis)
    model_energy = tf.keras.Model(
        inputs=M_in, outputs=M_lincomb, name="linear_basis_energy"
    )

    return model_energy


def model_e_diff(beta, model_energy):

    M_in_concat = [
        tf.keras.layers.Input(shape=(None, None)),
        tf.keras.layers.Input(shape=(None, None)),
    ]
    M_energy_diff = tf.keras.layers.Subtract(name="energy_diff")(
        [model_energy(M_in_concat[1]), model_energy(M_in_concat[0])]
    )
    M_exp = tf.keras.layers.Lambda(lambda x: tf.exp(-beta * x), name="exp")(
        M_energy_diff
    )
    model = tf.keras.Model(inputs=M_in_concat, outputs=M_exp, name="exp_energy_diff")

    return model


class ModelGroup:
    def __init__(self, config_ising, config_train, energy=None):
        if not energy:
            self.energy = deep_conv_e(
                config_train["conv_activation"],
                config_train["nfilters"],
                config_train["kernel_size"],
                config_train["dense_nodes"],
                config_train["dense_activation"],
            )
        else:
            self.energy = energy
        self.ediff = model_e_diff(config_ising["beta"], self.energy)

