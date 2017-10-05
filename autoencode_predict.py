import tensorflow as tf
from autoencode_model import Model


class predict:

    def __init__(self, name="meta-data/mnist/autoencode_model", color_depth=1):
        self.name = name
        self.color_depth = color_depth
        self.sess = None

    def start(self):
        if self.sess is None:
            tf.reset_default_graph()
            self.autoencode_model = Model(color_depth=self.color_depth)
            self.sess = tf.Session()

    def stop(self):
        if self.sess is not None:
            self.autoencode_model = None
            self.sess.close()
            self.sess = None
            tf.reset_default_graph()

    def reset(self):
        self.start()
        print "Resetting session ...",
        self.sess.run(tf.global_variables_initializer())
        print "done."

    def save(self):
        self.start()
        print "Saving session ...",
        tf.train.Saver().save(self.sess, self.name)
        print "done."

    def restore(self):
        self.start()
        print "Restoring session ...",
        try:
            tf.train.Saver().restore(self.sess, self.name)
        except:
            print "Can't restore. Resetting ...",
            self.reset()
        print "done."

    def doEpochOfTraining(self, loss, train, data_feed, batches=0, batch_size=100):
        batches = batches if batches > 0 else data_feed.getImages().shape[0] / batch_size
        for index in range(1, batches + 1):
            result, _ = self.sess.run([loss, train], feed_dict={self.autoencode_model.x_in: data_feed.nextBatch(
                batch_size)})
            if index == 1 or index == batches:
                print "index :", index, ", loss:", result
