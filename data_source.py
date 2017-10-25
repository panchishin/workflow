import numpy as np


class Mnist:

    def __init__(self, training=True):
        self.training = training

    def init(self):
        self.get_mnist_data(self.training)

    def get_mnist_data(self, training):
        from tensorflow.examples.tutorials.mnist import input_data
        import workflow_util

        with workflow_util.block_stdout():
            mnist = input_data.read_data_sets('./cache', one_hot=True)

        self.images = mnist.train.images if training else mnist.test.images
        self.labels = mnist.train.labels if training else mnist.test.labels

    def getImages(self):
        return self.images

    def getLabels(self):
        return self.labels


class FileReader:

    def __init__(self, files, label_names, height=240, width=240, channels=3):

        self.files = files
        self.label_names = label_names
        self.height = height
        self.width = width
        self.channels = channels

    def init(self):
        import numpy as np

        one_hot, identity = self._initOneHot(self.label_names)

        self.labels = []
        self.images = []
        self._preImageFetch()
        for file_name, label in zip(self.files, self.label_names):
            try:
                for image in self.getImagesFromFile(file_name):
                    self.labels.append(identity[one_hot.index(label), :].tolist())
                    self.images.append(image)
            except:
                print "file", file_name, "not found"
                pass

        self.labels = np.array(self.labels)
        self.images = np.array(self.images)

        self._postImageFetch()

    def _initOneHot(self, label_names):
        import numpy as np
        one_hot = []
        for label in label_names:
            if label not in one_hot:
                one_hot.append(label)
        identity = np.identity(len(one_hot), dtype=int)
        return one_hot, identity

    def _preImageFetch(self):
        import tensorflow as tf

        self.tf_img_name = tf.placeholder(dtype=tf.string)
        self.tf_img = tf.image.decode_jpeg(tf.read_file(self.tf_img_name), channels=self.channels)
        self.tf_img = tf.image.resize_image_with_crop_or_pad(self.tf_img, self.height, self.width)
        self.tf_img = tf.cast(self.tf_img, dtype=tf.float32)
        self.tf_img = self.tf_img / 256.0
        self.tf_img_flip = tf.image.flip_left_right(self.tf_img)
        self.sess = tf.Session()

    def _postImageFetch(self):
        self.sess.close()
        del(self.sess)

    def nameToURL(self, name):
        BASE_URL = 'https://mygardenorg.s3.amazonaws.com/plantifier/'
        return BASE_URL + name

    def getImagesFromFile(self, file_name):
        import os
        import urllib
        full_name = "data/" + file_name
        if not os.path.isfile(full_name):
            print "downloading",self.nameToURL(file_name),"as files", file_name
            urllib.urlretrieve(self.nameToURL(file_name), full_name)
        images = self.sess.run([self.tf_img, self.tf_img_flip], feed_dict={self.tf_img_name: full_name})
        return images

    def getImages(self):
        return self.images

    def getLabels(self):
        return self.labels


class ConcatWrapper:

    def __init__(self, sources):
        self.sources = sources
        self.images = None
        self.labels = None

    def init(self):
        self.concat(self.sources)
        del self.sources

    def concat(self, sources):
        self.images = np.concatenate([source.getImages() for source in self.sources], 0)
        self.labels = np.concatenate([source.getLabels() for source in self.sources], 0)

    def getImages(self):
        return self.images

    def getLabels(self):
        return self.labels


class SliceWrapper:

    def __init__(self, source, width=64, stride=18):
        self.source = source
        self.width = width
        self.stride = stride
        self.images = None
        self.labels = None

    def init(self):
        self.source.init()
        self.set(self.source)
        del self.source

    def set(self, source):
        source_images = source.getImages()
        source_labels = source.getLabels()
        x_range = range(0, source_images.shape[1] - self.width + 1, self.stride)
        y_range = range(0, source_images.shape[2] - self.width + 1, self.stride)

        for x in x_range:
            for y in y_range:
                sub_images = source_images[:, x:(x + self.width), y:(y + self.width), :]
                sub_labels = source_labels
                if self.images is None:
                    self.images = sub_images
                    self.labels = sub_labels
                else:
                    self.images = np.concatenate((self.images, sub_images), 0)
                    self.labels = np.concatenate((self.labels, sub_labels), 0)

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
