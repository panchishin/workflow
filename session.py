import tensorflow as tf
from autoencode_model import Model
autoencode_model = Model()

LEARNING_RATE = 1e-3

sess = None

def startSession() :
  global sess
  if sess == None :
    sess = tf.Session()

def resetSession() :
  print "Resetting session ..."
  sess.run( tf.global_variables_initializer() )
  print "... done."

def saveSession() :
  print "Saving session ..."
  tf.train.Saver().save(sess,"meta-data/autoencode_model")
  print "... done."

def restoreSession() :
  startSession()
  print "Restoring session ..."
  try :
    tf.train.Saver().restore(sess,"meta-data/autoencode_model")
  except :
    print "Can't restore.  Resetting."
    resetSession()
  print "... done."


def doEpochOfTraining( loss, train, data_feed, batches=55000/100, batch_size=100, rate=LEARNING_RATE ) :
  for index in range(1,batches+1) :
    result,_ = sess.run( [loss,train], feed_dict={autoencode_model.x_in:data_feed.nextBatch(batch_size),autoencode_model.learning_rate:rate})
    if index == 1 or index == batches :
        print "index :",index,", loss:", result


