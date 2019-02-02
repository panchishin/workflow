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


def imageloss(target,result):
    # visually appeasing L1 & L2 blended loss from https://arxiv.org/pdf/1511.08861.pdf
    diff = target - result
    return 0.16 * tf.abs(diff) + 0.84 * tf.square(diff)


def autoencode(input, target, depth, color_depth, top_k, reuse=True):
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

    loss = imageloss(target,result)
    loss = tf.reduce_mean(loss,[1,2,3])

    loss, top_result = tf.nn.top_k(loss,k=top_k)
    loss = tf.reduce_mean(loss)

    return result, loss, embedding, top_result


class Model:

    def __init__(self, size=32, high_low_noise_value=0.02, color_depth=1):
        self.SIZE = size
        self.HIGH_LOW_NOISE = high_low_noise_value
        self.COLOR_DEPTH = color_depth

        self.top_k = tf.placeholder_with_default(1,[], name="top_k")
        self.x_in = tf.placeholder(tf.float32, [None, size, size, color_depth], name="x0")

        self.x_noisy = layer.high_low_noise(self.x_in, high_low_noise_value)

        self.x_out_5, self.loss_5, self.embedding, self.top_result = autoencode(self.x_noisy, self.x_in, 5, color_depth, self.top_k, False)
        self.x_out_4, self.loss_4, _, _ = autoencode(self.x_noisy, self.x_in, 4, color_depth, self.top_k)
        self.x_out_3, self.loss_3, _, _ = autoencode(self.x_noisy, self.x_in, 3, color_depth, self.top_k)
        self.x_out_2, self.loss_2, _, _ = autoencode(self.x_noisy, self.x_in, 2, color_depth, self.top_k)
        self.x_out_1, self.loss_1, _, _ = autoencode(self.x_noisy, self.x_in, 1, color_depth, self.top_k)

        self.loss_5 = ( self.loss_5 + self.loss_4 + self.loss_3 + self.loss_2 + self.loss_1 ) / 5.0

        self.train_5 = tf.train.AdamOptimizer().minimize(self.loss_5)
