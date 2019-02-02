import tensorflow as tf
from autoencode_model import Model
import numpy as np
from time import time

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

    def doEpochOfTraining(self, loss, train, data_feed, batches=0, batch_size=1024, top_k=32, elapse=1):
        start_time = time()
        batches = batches if batches > 0 else data_feed.getImages().shape[0] / batch_size
        result = []
        index = 0.
        while time() < start_time + elapse :
            index += 1.

            next_data = data_feed.nextBatch(batch_size)
            top_k = min(16,next_data.shape[0])

            top_result = self.sess.run(self.autoencode_model.top_result, feed_dict={ 
                self.autoencode_model.x_in: next_data,
                self.autoencode_model.top_k: top_k
                })
            loss_result, _ = self.sess.run([loss, train], feed_dict={ 
                self.autoencode_model.x_in: next_data[top_result],
                self.autoencode_model.top_k: len(top_result)
                })

        print "  loss", loss_result.shape, np.mean(loss_result), ", sec/batch",(time()-start_time)/index

        return result
