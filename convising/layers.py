import tensorflow as tf


class PeriodicPad2D(tf.keras.layers.Layer):
    def __init__(self, pad_size, **kwargs):
        self.pad_size = pad_size
        super(PeriodicPad2D, self).__init__(**kwargs)

    def build(self, input_shapes):
        super(PeriodicPad2D, self).build(input_shapes)

    def call(self, inputs):
        leftpad = int(self.pad_size / 2)
        rightpad = self.pad_size - leftpad
        x = inputs
        x_1 = tf.concat([x[:, :, -leftpad:], x, x[:, :, :rightpad]], axis=2)
        y = tf.concat([x_1[:, -leftpad:, :], x_1, x_1[:, :rightpad, :]], axis=1)

        return tf.expand_dims(y, axis=-1)

    def compute_output_shape(self, input_shapes):
        return (input_shapes[0], None, None, 1)


class NFSym(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(NFSym, self).__init__(**kwargs)

    def build(self, input_shapes):
        super(NFSym, self).build(input_shapes)

    def call(self, inputs):
        return tf.map_fn(lambda x: x[0, 0] * x, inputs)

    def compute_output_shape(self, input_shapes):
        return input_shapes


class ConvBias(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(ConvBias, self).__init__(**kwargs)

    def build(self, input_shapes):
        self.bias = self.add_weight(
            name="bias", shape=(input_shapes[3],), initializer="zeros", trainable=True
        )
        super(ConvBias, self).build(input_shapes)

    def call(self, inputs):
        return tf.bias_add(inputs, self.bias)

    def compute_output_shape(self, input_shapes):
        return input_shapes


class LinearBasis(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(LinearBasis, self).__init__(**kwargs)

    def build(self, input_shape):
        super(LinearBasis, self).build(input_shape)

    def call(self, x):
        first_nearest = tf.expand_dims(
            tf.math.reduce_sum(
                0.5
                * (
                    x * tf.roll(x, 1, axis=1)
                    + x * tf.roll(x, -1, axis=1)
                    + x * tf.roll(x, 1, axis=2)
                    + x * tf.roll(x, -1, axis=2)
                ),
                axis=(1, 2),
            ),
            axis=1,
        )
        second_nearest = tf.expand_dims(
            tf.math.reduce_sum(
                0.5
                * (
                    x * tf.roll(x, (1, 1), axis=(1, 2))
                    + x * tf.roll(x, (1, -1), axis=(1, 2))
                    + x * tf.roll(x, (-1, 1), axis=(1, 2))
                    + x * tf.roll(x, (-1, -1), axis=(1, 2))
                ),
                axis=(1, 2),
            ),
            axis=1,
        )
        four_spins = tf.expand_dims(
            tf.math.reduce_sum(
                x
                * tf.roll(x, 1, axis=1)
                * tf.roll(x, 1, axis=2)
                * tf.roll(x, (1, 1), axis=(1, 2)),
                axis=(1, 2),
            ),
            axis=1,
        )
        return tf.concat([first_nearest, second_nearest, four_spins], 1)

    def compute_output_shape(self, input_shape):
        return (input_shape[0], 3)


class GlobalD4(tf.keras.constraints.Constraint):
    def __call__(self, weights):
        w = tf.squeeze(weights)
        w_rot1 = tf.image.rot90(w, k=1)
        w_rot2 = tf.image.rot90(w, k=2)
        w_rot3 = tf.image.rot90(w, k=3)
        w_horiz = w[:, ::-1, :]
        w_vert = w[::-1, :, :]
        w_diag1 = w_rot1[::-1, :, :]
        w_diag2 = w_rot1[:, ::-1, :]
        w = w + w_rot1 + w_rot2 + w_rot3 + w_horiz + w_vert + w_diag1 + w_diag2
        out_w = tf.expand_dims(tf.divide(w, 8.0), axis=2)
        return out_w
