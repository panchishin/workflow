import tensorflow as tf
from autoencode_model import Model


class predict:

    def __init__(self, name="meta-data/autoencode_model", color_depth=1):
        self.name = name
        self.autoencode_model = Model(color_depth=color_depth)
        self.LEARNING_RATE = 1e-3
        self.sess = None

    def start(self):
        if self.sess is None:
            self.sess = tf.Session()

    def reset(self):
        print "Resetting session ...",
        self.sess.run(tf.global_variables_initializer())
        print "done."

    def save(self):
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

    def doEpochOfTraining(self, loss, train, data_feed, batches=0, batch_size=100, rate=None):
        rate = self.LEARNING_RATE if rate is None else rate
        batches = batches if batches > 0 else data_feed.getImages().shape[0] / batch_size
        for index in range(1, batches + 1):
            result, _ = self.sess.run([loss, train], feed_dict={self.autoencode_model.x_in: data_feed.nextBatch(
                batch_size), self.autoencode_model.learning_rate: rate})
            if index == 1 or index == batches:
                print "index :", index, ", loss:", result
