class Mnist:
    def __init__(self, training=True):
        self.training = training

    def init(self):
        self.get_mnist_data(self.training)

    def get_mnist_data(self, training):
        from tensorflow.examples.tutorials.mnist import input_data
        mnist = input_data.read_data_sets('./cache', one_hot=True)
        self.images = mnist.train.images if training else mnist.test.images
        self.labels = mnist.train.labels if training else mnist.test.labels

    def getImages(self):
        return self.images

    def getLabels(self):
        return self.labels


class ReshapeWrapper:
    def __init__(self, source, target_shape):
        self.source = source
        self.target_shape = target_shape

    def init(self):
        self.source.init()
        self.set(self.source, self.target_shape)
        del self.source
        del self.target_shape

    def set(self, source, target_shape):
        self.images = source.getImages().reshape([source.getImages().shape[0]] + target_shape)
        self.labels = source.getLabels()

    def getImages(self):
        return self.images

    def getLabels(self):
        return self.labels


class ResizeWrapper:
    def __init__(self, source, target_size):
        self.source = source
        self.target_size = target_size

    def init(self):
        self.source.init()
        self.resize(self.source, self.target_size)
        del self.source
        del self.target_size

    def resize(self, source, target_size):
        self.labels = source.getLabels()

        import tensorflow as tf
        image_in = tf.placeholder(tf.float32, [None, None, None, None])
        image_out = tf.image.resize_images(image_in, target_size, method=tf.image.ResizeMethod.NEAREST_NEIGHBOR,
                                           align_corners=False)
        with tf.Session() as sess:
            self.images = sess.run(image_out, feed_dict={image_in: source.getImages()})

    def getImages(self):
        return self.images

    def getLabels(self):
        return self.labels


class BatchWrapper:
    def __init__(self, source):
        self.source = source

    def init(self):
        self.source.init()
        self.shuffle()

    def shuffle(self):
        import numpy as np
        self.index = range(self.source.getImages().shape[0])
        np.random.shuffle(self.index)
        self.offset = 0

    def nextBatch(self, size):
        if self.offset + size > self.source.getImages().shape[0]:
            self.shuffle()
        result = self.source.getImages()[self.index[self.offset:self.offset + size]]
        self.offset += size
        return result

    def getImages(self):
        return self.source.getImages()

    def getLabels(self):
        return self.source.getLabels()


class LazyLoadWrapper:
    def __init__(self, source):
        self.source = source
        self.uninitialized = True

    def init(self):
        if self.uninitialized:
            self.source.init()
            self.uninitialized = False

    def nextBatch(self, size):
        self.init()
        return self.source.nextBatch(size)

    def getImages(self):
        self.init()
        return self.source.getImages()

    def getLabels(self):
        self.init()
        return self.source.getLabels()


if __name__ == '__main__':
    print "RUNNING TESTS"
    mnist = LazyLoadWrapper(Mnist())
    if mnist.getImages().shape == tuple([55000, 28 * 28]):
        print ".",
    else:
        print "FAIL", mnist.getImages().shape, "should equal [55000,28*28]"

    reshaped = LazyLoadWrapper(ReshapeWrapper(mnist, [28, 28, 1]))
    if reshaped.getImages().shape == tuple([55000, 28, 28, 1]):
        print ".",
    else:
        print "FAIL", reshaped.getImages().shape, "should equal [55000,28,28,1]"

    resized = LazyLoadWrapper(ResizeWrapper(reshaped, [32, 32]))
    if resized.getImages().shape == tuple([55000, 32, 32, 1]):
        print ".",
    else:
        print "FAIL", resized.getImages().shape, "should equal [55000,32,32,1]"

    batch = BatchWrapper(resized)
    batch.init()
    if batch.nextBatch(10).shape == tuple([10, 32, 32, 1]):
        print ".",
    else:
        print "FAIL", resized.getImages().shape, "should equal [10,32,32,1]"

    print "done"
