from keras.models import Model
from keras.engine.topology import Layer
from keras import backend as K
from keras.constraints import Constraint
from keras.layers.convolutional import _Conv

import tensorflow as tf


class PeriodicPad2D(Layer):

    def __init__(self, pad_size, **kwargs):
        self.pad_size = pad_size
        super(PeriodicPad2D, self).__init__(**kwargs)

    def build(self, input_shapes):
        super(PeriodicPad2D, self).build(input_shapes)

    def call(self, inputs):
        leftpad = int(self.pad_size/2)
        rightpad = self.pad_size - leftpad
        x = inputs
        x_1 = K.concatenate([x[:,:,-leftpad:], x, x[:,:,:rightpad]], axis=2)
        y = K.concatenate([x_1[:,-leftpad:,:], x_1, x_1[:,:rightpad,:]], axis=1)

        return K.expand_dims(y)

    def compute_output_shape(self, input_shapes):
        return (input_shapes[0], None, None, 1)


class NFSym(Layer):

    def __init__(self, **kwargs):
        super(NFSym, self).__init__(**kwargs)

    def build(self, input_shapes):
        super(NFSym, self).build(input_shapes)

    def call(self, inputs):
        return tf.map_fn(lambda x: x[0,0]*x, inputs)

    def compute_output_shape(self, input_shapes):
        return input_shapes


class ConvBias(Layer):

    def __init__(self, **kwargs):
        super(ConvBias, self).__init__(**kwargs)

    def build(self, input_shapes):
        self.bias = self.add_weight(name='bias', shape=(input_shapes[3],), initializer='zeros', trainable=True)
        super(ConvBias, self).build(input_shapes)

    def call(self, inputs):
        return K.bias_add(inputs, self.bias)

    def compute_output_shape(self, input_shapes):
        return input_shapes


class LinearBasis(Layer):

    def __init__(self, **kwargs):
        super(LinearBasis, self).__init__(**kwargs)

    def build(self, input_shape):
        super(LinearBasis, self).build(input_shape)

    def call(self, x):
        first_nearest = K.expand_dims(K.sum(0.5 * (x * tf.manip.roll(x, 1, axis=1) + x * tf.manip.roll(x, -1, axis=1) + x * tf.manip.roll(x, 1, axis=2) + x * tf.manip.roll(x, -1, axis=2)), axis=(1,2)), axis=1)
        second_nearest = K.expand_dims(K.sum(0.5 * (x * tf.manip.roll(x, (1,1), axis=(1,2)) + x * tf.manip.roll(x, (1,-1), axis=(1,2)) + x * tf.manip.roll(x, (-1,1), axis=(1,2)) + x * tf.manip.roll(x, (-1,-1), axis=(1,2))), axis=(1,2)), axis=1)
        four_spins = K.expand_dims(K.sum(x * tf.manip.roll(x, 1, axis=1) * tf.manip.roll(x, 1, axis=2) * tf.manip.roll(x, (1,1), axis=(1,2)), axis=(1,2)), axis=1)
        return tf.concat([first_nearest, second_nearest, four_spins], 1)

    def compute_output_shape(self, input_shape):
        return (input_shape[0], 3)


class GlobalD4(Constraint):

    def __call__(self, weights):
        w = tf.squeeze(weights)
        w_rot1 = tf.image.rot90(w,k=1)
        w_rot2 = tf.image.rot90(w,k=2)
        w_rot3 = tf.image.rot90(w,k=3)
        w_horiz = w[:,::-1,:]
        w_vert = w[::-1,:,:]
        w_diag1 = w_rot1[::-1,:,:]
        w_diag2 = w_rot1[:,::-1,:]
        w = w + w_rot1 + w_rot2 + w_rot3 + w_horiz + w_vert + w_diag1 + w_diag2
        out_w = K.expand_dims(tf.divide(w,8.0), axis=2)
        return out_w


class Conv2DNFSym(_Conv):

    def __init__(self, filters,
                 kernel_size,
                 strides=(1, 1),
                 padding='valid',
                 data_format=None,
                 dilation_rate=(1, 1),
                 activation=None,
                 use_bias=True,
                 kernel_initializer='glorot_uniform',
                 bias_initializer='zeros',
                 kernel_regularizer=None,
                 bias_regularizer=None,
                 activity_regularizer=None,
                 kernel_constraint=None,
                 bias_constraint=None,
                 nfsym='none',
                 **kwargs):
        if nfsym == 'd4' or nfsym == 'all':
            kernel_constraint = GlobalD4()
        super(Conv2DNFSym, self).__init__(
            rank=2,
            filters=filters,
            kernel_size=kernel_size,
            strides=strides,
            padding=padding,
            data_format=data_format,
            dilation_rate=dilation_rate,
            activation=activation,
            use_bias=use_bias,
            kernel_initializer=kernel_initializer,
            bias_initializer=bias_initializer,
            kernel_regularizer=kernel_regularizer,
            bias_regularizer=bias_regularizer,
            activity_regularizer=activity_regularizer,
            kernel_constraint=kernel_constraint,
            bias_constraint=bias_constraint,
            **kwargs)

    def get_config(self):
        conv_config = super(Conv2DNFSym, self).get_config()
        conv_config.pop('rank')
        return conv_config
