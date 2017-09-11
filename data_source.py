import numpy as np


class Mnist:

  def __init__(self,training=True):
    self.get_mnist_data(training)

  def get_mnist_data(self,training) :
    from tensorflow.examples.tutorials.mnist import input_data
    mnist = input_data.read_data_sets('./cache', one_hot=True)
    self.images = mnist.train.images if training else mnist.test.images
    self.labels = mnist.train.labels if training else mnist.test.labels

  def getImages(self):
    return self.images

  def getLabels(self):
    return self.labels


class Reshape:

  def __init__(self, source, target_shape):
    self.set(source, target_shape)

  def set(self, source, target_shape) :
    self.images = source.getImages().reshape( [source.getImages().shape[0]] + target_shape )
    self.labels = source.getLabels()
    
  def getImages(self):
    return self.images

  def getLabels(self):
    return self.labels


class Resize:

  def __init__(self, source, target_size):
    self.resize(source, target_size)

  def resize(self, source, target_size) :
    self.labels = source.getLabels()

    import tensorflow as tf
    image_in = tf.placeholder(tf.float32, [None, None, None, None])
    image_out = tf.image.resize_images( image_in, target_size, method=tf.image.ResizeMethod.NEAREST_NEIGHBOR, align_corners=False )
    with tf.Session() as sess :
      self.images = sess.run( image_out , feed_dict={ image_in : source.getImages() } )
    
  def getImages(self):
    return self.images

  def getLabels(self):
    return self.labels


class BatchWrapper:

  def __init__(self, source):
    self.source = source
    self.shuffle()

  def shuffle(self):
    self.index = range(self.source.getImages().shape[0])
    np.random.shuffle(self.index)
    self.offset = 0

  def getBatch(self,size):
    if self.offset + size > self.source.getImages().shape[0] :
      self.shuffle()
    result = self.source.getImages()[ self.index[ self.offset:self.offset+size] ]
    self.offset += size
    return result

  def getImages(self):
    return self.source.getImages()

  def getLabels(self):
    return self.source.getLabels()




if __name__ == '__main__' :
  print "RUNNING TESTS"
  mnist = Mnist()
  if mnist.getImages().shape == tuple([55000,28*28]) :
    print ".",
  else :
    print "FAIL", mnist.getImages().shape, "should equal [55000,28*28]"

  reshaped = Reshape( mnist , [28,28,1] )
  if reshaped.getImages().shape == tuple([55000,28,28,1]) :
    print ".",
  else :
    print "FAIL", reshaped.getImages().shape, "should equal [55000,28,28,1]"

  resized = Resize( reshaped, [32,32] )
  if resized.getImages().shape == tuple([55000,32,32,1]) :
    print ".",
  else :
    print "FAIL", resized.getImages().shape, "should equal [55000,32,32,1]"

  batch = BatchWrapper( resized )
  if batch.getBatch(10).shape == tuple([10,32,32,1]) :
    print ".",
  else :
    print "FAIL", resized.getImages().shape, "should equal [10,32,32,1]"

  print "done"

