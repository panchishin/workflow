import model

model.restoreSession()


def get_mnist_data() :
  from tensorflow.examples.tutorials.mnist import input_data
  return input_data.read_data_sets('./cache', one_hot=True)

mnist = get_mnist_data()


if __name__ == "__main__" :
  print "Start training test ..."
  model.doEpochOfTraining( model.loss_6 , model.train_6 , data_feed=mnist.train, batches=10 , batch_size=50 )
  print "... finished training test."
  model.saveSession()
  exit()


