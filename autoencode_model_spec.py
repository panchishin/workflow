import autoencode_model
import session

session.restoreSession()


def get_mnist_data() :
  from tensorflow.examples.tutorials.mnist import input_data
  return input_data.read_data_sets('./cache', one_hot=True)

mnist = get_mnist_data()


print "Start training test ..."
session.doEpochOfTraining( autoencode_model.loss_6 , autoencode_model.train_6 , data_feed=mnist.train, batches=10 , batch_size=50 )
print "... finished training test."
session.saveSession()
exit()


