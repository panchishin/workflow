import tensorflow as tf
import layer

def encode(image, layers_in, layers_out=0, width=5, reuse=True):
    with tf.variable_scope("conv" + str(layers_in), reuse=reuse):
        layers_out = layers_in * 2 if layers_out == 0 else layers_out
        image = layer.conv(image, layers_in, layers_out, stride=2, width=width, name="stage1")
        image = tf.tanh(image)
        return image


def decode(image, layers_in, layers_out=0, width=5, reuse=True):
    with tf.variable_scope("deconv" + str(layers_in), reuse=reuse):
        layers_out = layers_in / 2 if layers_out == 0 else layers_out
        image = layer.upscaleFlat(image, scale=2)
        image = layer.conv(image, layers_in, layers_out, width=width, name="stage1")
        logits = image
        image = tf.tanh(image)
        return image , logits


def autoencode(input, target, depth, color_depth, reuse=True):
    autoencoding_layer = [input]
    for index in range(depth):
        width = [7,7,7,5,3][index]
        autoencoding_layer.append(encode(autoencoding_layer[-1], color_depth * 2 ** index, width=width, reuse=reuse))
    embedding = autoencoding_layer[-1]
    logits = None
    for index in range(depth, 0, -1):
        width = [7,7,7,5,3][index-1]
        image, logits = decode(autoencoding_layer[-1], color_depth * 2 ** index, width=width, reuse=reuse)
        autoencoding_layer.append(image)
    result = tf.sigmoid(logits)

    # visually appeasing L1 & L2 blended loss from https://arxiv.org/pdf/1511.08861.pdf
    loss = tf.reduce_mean( 0.16 * tf.abs(target - result) + 0.84 * tf.square(target - result) )

    return result, loss, embedding


class Model:

    def __init__(self, size=32, high_low_noise_value=0.02, color_depth=1):
        self.SIZE = size
        self.HIGH_LOW_NOISE = high_low_noise_value
        self.COLOR_DEPTH = color_depth

        self.x_in = tf.placeholder(tf.float32, [None, size, size, color_depth], name="x0")

        self.x_noisy = layer.high_low_noise(self.x_in, high_low_noise_value)

        self.x_out_5, self.loss_5, self.embedding = autoencode(self.x_noisy, self.x_in, 5, color_depth, False)
        self.x_out_4, self.loss_4, _ = autoencode(self.x_noisy, self.x_in, 4, color_depth)
        self.x_out_3, self.loss_3, _ = autoencode(self.x_noisy, self.x_in, 3, color_depth)
        self.x_out_2, self.loss_2, _ = autoencode(self.x_noisy, self.x_in, 2, color_depth)
        self.x_out_1, self.loss_1, _ = autoencode(self.x_noisy, self.x_in, 1, color_depth)

        self.loss_6 = self.loss_5
        self.loss_5 = ( self.loss_5 + self.loss_4 + self.loss_3 + self.loss_2 + self.loss_1 ) / 5.0
        self.loss_4 = ( self.loss_4 + self.loss_3 + self.loss_2 + self.loss_1 ) / 4.0
        self.loss_3 = ( self.loss_3 + self.loss_2 + self.loss_1 ) / 3.0
        self.loss_2 = ( self.loss_2 + self.loss_1 ) / 2.0

        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            self.train_1 = tf.train.AdamOptimizer().minimize(self.loss_1)
            self.train_2 = tf.train.AdamOptimizer().minimize(self.loss_2)
            self.train_3 = tf.train.AdamOptimizer().minimize(self.loss_3)
            self.train_4 = tf.train.AdamOptimizer().minimize(self.loss_4)
            self.train_5 = tf.train.AdamOptimizer().minimize(self.loss_5)
            self.train_6 = tf.train.AdamOptimizer().minimize(self.loss_6)
