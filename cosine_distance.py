import layer
import model
import session
import numpy as np
import embeddings
import tensorflow as tf
session.restoreSession()


def get_mnist_data() :
  from tensorflow.examples.tutorials.mnist import input_data
  return input_data.read_data_sets('./cache', one_hot=True)

mnist = get_mnist_data()
embeddings.data_set = mnist.test.images

#session.doEpochOfTraining( model.loss_6, model.train_6, mnist.train )
#session.saveSession()

numbers = [3, 2, 1, 18, 4, 8, 11, 0, 61, 7]

print "The numbers are",np.argmax( mnist.test.labels[numbers] , 1 )
print "And their indexes are",numbers

print "Pretend labeling the first 10 of each embedding ..."
data_in = []
label_in = []
for index in numbers:
  nearest = embeddings.nearestNeighbourByIndex(index)
  result = zip( mnist.test.labels[ nearest ] , nearest )
  for label,data_index in result :
    label_in.append( label )
    data_in.append( embeddings.getEmbeddings()[data_index] )
print "... done"

data_in = np.array(data_in)
label_in = np.array(label_in)

emb_in = tf.placeholder(tf.float32, [None, 32] , name="emb_in")
category_in = tf.placeholder(tf.float32, [None, 10] , name="category")
dropout = tf.placeholder(tf.float32)

layer0 = layer.fully_connected( emb_in, 32 , 32 , "layer1" )
# TODO ADD A NON LINEAR LAYER HERE AND TEST
layer1 = tf.nn.dropout( tf.nn.tanh(layer0) , dropout )
layer2 = layer.fully_connected( layer1, 32 , 10 , "layer2" )
category_out = tf.nn.sigmoid( layer2 )

loss = tf.reduce_mean( tf.nn.softmax_cross_entropy_with_logits( labels=category_in, logits=layer2) )
train = tf.train.AdamOptimizer(1e-3).minimize(loss)

correct = tf.reduce_mean( tf.cast( tf.equal( tf.argmax( category_in , 1 ) , tf.argmax( layer2 , 1 ) ) , dtype=tf.float32 ) )

sess = tf.Session()


sess.run( tf.global_variables_initializer() )

target_correct = 0.97
for _ in range(20) :
  for iteration in range(250) :
    sess.run(train,feed_dict={emb_in:data_in,category_in:label_in,dropout:.5})
    percent_correct = sess.run(correct,feed_dict={emb_in:data_in,category_in:label_in,dropout:1.0})
    if percent_correct >= target_correct :
      break
  print "Loss =", sess.run(loss,feed_dict={emb_in:data_in,category_in:label_in,dropout:.5}),
  print " selected correct =", round( 100*percent_correct , 2),"%"
  if percent_correct >= target_correct :
   break

print " full set correct =", round( 100*sess.run(correct,feed_dict={emb_in:embeddings.getEmbeddings(),category_in:mnist.test.labels,dropout:1.0}) , 2),"%"

threshold = sess.run( category_out, feed_dict={emb_in:embeddings.getEmbeddings(),category_in:mnist.test.labels,dropout:1.0})
average_threshold = np.mean( np.max( threshold , 1 ) )
print "The average max threshold is", average_threshold

high_thresholds = np.max( threshold , 1 ) > average_threshold
print "Using high_threshold of",average_threshold
print "There are",np.sum(high_thresholds),"high thresholds"
print "high_thresholds correct =", round( 100*sess.run(correct,feed_dict={emb_in:embeddings.getEmbeddings()[high_thresholds],category_in:mnist.test.labels[high_thresholds],dropout:1.0}) , 2),"%"

average_threshold = 1.-(1.-average_threshold)/2

high_thresholds = np.max( threshold , 1 ) > average_threshold
print "Using high_threshold of",average_threshold
print "There are",np.sum(high_thresholds),"high thresholds"
print "high_thresholds correct =", round( 100*sess.run(correct,feed_dict={emb_in:embeddings.getEmbeddings()[high_thresholds],category_in:mnist.test.labels[high_thresholds],dropout:1.0}) , 2),"%"

average_threshold = 1.-(1.-average_threshold)/2

high_thresholds = np.max( threshold , 1 ) > average_threshold
print "Using high_threshold of",average_threshold
print "There are",np.sum(high_thresholds),"high thresholds"
print "high_thresholds correct =", round( 100*sess.run(correct,feed_dict={emb_in:embeddings.getEmbeddings()[high_thresholds],category_in:mnist.test.labels[high_thresholds],dropout:1.0}) , 2),"%"
