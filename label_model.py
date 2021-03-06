import tensorflow as tf
import layer


class model:

    def __init__(self, number_of_classes=2, training_factor=.3, width=32):
        self.width = width
        self.emb_in = tf.placeholder(tf.float32, [None, self.width], name="emb_in")
        self.category_in = tf.placeholder(tf.float32, [None, number_of_classes], name="category")
        self.dropout = tf.placeholder(tf.float32)

        with tf.variable_scope("layer0"):
            self.W0 = layer.weight_variable([self.width, self.width], name=("layer0_weight"))
            self.b0 = layer.bias_variable([self.width], name=("layer0_bias"))
            self.layer0 = tf.add(tf.matmul(self.emb_in, self.W0), self.b0)
            self.layer0 = tf.tanh(self.layer0)
            self.layer0 = tf.nn.dropout(self.layer0, self.dropout)

        with tf.variable_scope("layer1"):
            self.W1 = layer.weight_variable([self.width, number_of_classes], name=("layer1_weight"))
            self.b1 = layer.bias_variable([number_of_classes], name=("layer1_bias"))
            self.layer1 = tf.add(tf.matmul(self.layer0, self.W1), self.b1)

        self.category_out = tf.nn.softmax(self.layer1)

        self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=self.category_in, logits=self.layer1))
        self.train = tf.train.AdamOptimizer(training_factor).minimize(self.loss)

        self.error = 1.0 - tf.reduce_mean(
            tf.cast(tf.equal(tf.argmax(self.category_in, 1), tf.argmax(self.layer1, 1)), dtype=tf.float32))
