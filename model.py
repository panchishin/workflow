import tensorflow as tf
import numpy as np
import layer

SIZE = 32
LEARNING_RATE = 1e-2
HIGH_LOW_NOISE = 0.02


x0 = tf.placeholder(tf.float32, [None, 28*28] , name="x0")
learning_rate = tf.placeholder( tf.float32 )

x_reshape = tf.reshape( x0, [-1,28,28,1], name="x_in" )
x_enlarge = tf.image.resize_images( x_reshape, [SIZE,SIZE], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR, align_corners=False )
x_in = x_enlarge
x_noisy = layer.high_low_noise( x_in , HIGH_LOW_NOISE)


def encode( image, layers_in, layers_out=0, width=3, reuse=True ) :
  with tf.variable_scope( "conv"+str(layers_in) , reuse=reuse ) :
    layers_out = layers_in * 2 if layers_out == 0 else layers_out
    image = layer.conv_relu( image , layers_in , layers_out , stride=2, width=width , name="stage1" )
    return image

def decode( image, layers_in, layers_out=0, width=3, reuse=True ) :
  with tf.variable_scope( "deconv"+str(layers_in) , reuse=reuse ) :
    layers_out = layers_in / 2 if layers_out == 0 else layers_out
    image = layer.upscaleFlat( image , scale=2 )
    image = layer.conv_relu( image , layers_in , layers_out , width=width , name="stage1" )
    return image


def autoencode(input,target,depth,reuse=True) :
  autoencoding_layer = [input]
  for index in range(depth) :
    autoencoding_layer.append( encode( autoencoding_layer[-1] , 2**index , reuse=reuse) )
  embedding = autoencoding_layer[-1]
  for index in range(depth,0,-1) :
    autoencoding_layer.append( decode( autoencoding_layer[-1] , 2**index , reuse=reuse) )
  result = autoencoding_layer[-1]
  loss = tf.log( tf.reduce_mean( tf.square( target - result ) ) )
  return result,loss,embedding


x_out_5,loss_5,conv5e = autoencode(x_noisy,x_in,5,False)
x_out_4,loss_4,_ = autoencode(x_noisy,x_in,4)
x_out_3,loss_3,_ = autoencode(x_noisy,x_in,3)
x_out_2,loss_2,_ = autoencode(x_noisy,x_in,2)
x_out_1,loss_1,_ = autoencode(x_noisy,x_in,1)

loss_6 = loss_5 + loss_4 + loss_3 + loss_2 + loss_1

update_ops   = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
with tf.control_dependencies(update_ops):
  train_1 = tf.train.AdamOptimizer(learning_rate).minimize(loss_1)
  train_2 = tf.train.AdamOptimizer(learning_rate).minimize(loss_2)
  train_3 = tf.train.AdamOptimizer(learning_rate).minimize(loss_3)
  train_4 = tf.train.AdamOptimizer(learning_rate).minimize(loss_4)
  train_5 = tf.train.AdamOptimizer(learning_rate).minimize(loss_5)
  train_6 = tf.train.AdamOptimizer(learning_rate).minimize(loss_6)


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
  tf.train.Saver().save(sess,"meta-data/model")
  print "... done."

def restoreSession() :
  startSession()
  print "Restoring session ..."
  try :
    tf.train.Saver().restore(sess,"meta-data/model")
  except :
    print "Can't restore.  Resetting."
    resetSession()
  print "... done."


def doEpochOfTraining( loss, train, data_feed, batches=55000/100, batch_size=100, rate=LEARNING_RATE ) :
  for index in range(1,batches+1) :
    result,_ = sess.run( [loss,train], feed_dict={x0:data_feed.next_batch(batch_size)[0],learning_rate:rate})
    if index == 1 or index == batches :
        print "index :",index,", loss:", result


