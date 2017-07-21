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


conv5a = encode( x_noisy , 1 , reuse=False)
conv5b = encode( conv5a , 2 , reuse=False )
conv5c = encode( conv5b , 4 , reuse=False )
conv5d = encode( conv5c , 8 , reuse=False )
conv5e = encode( conv5d , 16 , reuse=False )
deconv5a = decode( conv5e , 32 , reuse=False )
deconv5b = decode( deconv5a , 16 , reuse=False )
deconv5c = decode( deconv5b , 8 , reuse=False )
deconv5d = decode( deconv5c , 4 , reuse=False )
deconv5e = decode( deconv5d , 2 , reuse=False )

x_out_5 = deconv5e

conv4a = encode( x_noisy , 1 )
conv4b = encode( conv4a , 2 )
conv4c = encode( conv4b , 4 )
conv4d = encode( conv4c , 8 )
deconv4a = decode( conv4d , 16 )
deconv4b = decode( deconv4a , 8 )
deconv4c = decode( deconv4b , 4 )
deconv4d = decode( deconv4c , 2 )

x_out_4 = deconv4d

conv3a = encode( x_noisy , 1 )
conv3b = encode( conv3a , 2 )
conv3c = encode( conv3b , 4 )
deconv3a = decode( conv3c , 8 )
deconv3b = decode( deconv3a , 4 )
deconv3c = decode( deconv3b , 2 )

x_out_3 = deconv3c

conv2a = encode( x_noisy , 1 )
conv2b = encode( conv2a , 2 )
deconv2a = decode( conv2b , 4 )
deconv2b = decode( deconv2a , 2 )

x_out_2 = deconv2b

conv1a = encode( x_noisy , 1 )
deconv1a = decode( conv1a , 2 )

x_out_1 = deconv1a


loss_1_raw = tf.log( tf.reduce_mean( tf.square( x_in - x_out_1 ) ) )
loss_2_raw = tf.log( tf.reduce_mean( tf.square( x_in - x_out_2 ) ) )
loss_3_raw = tf.log( tf.reduce_mean( tf.square( x_in - x_out_3 ) ) )
loss_4_raw = tf.log( tf.reduce_mean( tf.square( x_in - x_out_4 ) ) )
loss_5_raw = tf.log( tf.reduce_mean( tf.square( x_in - x_out_5 ) ) )

loss_1 = loss_1_raw
loss_2 = loss_2_raw
loss_3 = loss_3_raw
loss_4 = loss_4_raw
loss_5 = loss_5_raw
loss_6 = loss_5_raw + loss_4_raw + loss_3_raw + loss_2_raw + loss_1_raw

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


