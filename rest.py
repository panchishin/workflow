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
    image = layer.conv_relu( image , layers_in , layers_out , stride=2, width=width )
    return image

def decode( image, layers_in, layers_out=0, width=3, reuse=True ) :
  with tf.variable_scope( "deconv"+str(layers_in) , reuse=reuse ) :
    layers_out = layers_in / 2 if layers_out == 0 else layers_out
    image = layer.upscaleFlat( image , scale=2 )
    image = layer.conv_relu( image , layers_in , layers_out , width=width )
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
loss_6 = loss_5_raw * 4 + loss_4_raw * 3 + loss_3_raw * 2 + loss_2_raw + loss_1_raw

update_ops   = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
with tf.control_dependencies(update_ops):
  train_1 = tf.train.AdamOptimizer(learning_rate).minimize(loss_1)
  train_2 = tf.train.AdamOptimizer(learning_rate).minimize(loss_2)
  train_3 = tf.train.AdamOptimizer(learning_rate).minimize(loss_3)
  train_4 = tf.train.AdamOptimizer(learning_rate).minimize(loss_4)
  train_5 = tf.train.AdamOptimizer(learning_rate).minimize(loss_5)
  train_6 = tf.train.AdamOptimizer(learning_rate).minimize(loss_6)


sess = tf.Session()

def resetSession() :
  sess.run( tf.global_variables_initializer() )

resetSession()





def get_mnist_data() :
  from tensorflow.examples.tutorials.mnist import input_data
  return input_data.read_data_sets('./cache', one_hot=True)

mnist = get_mnist_data()




def doEpochOfTraining( loss, train, batches=55000/100, batch_size=100, rate=LEARNING_RATE ) :
  for index in range(1,batches+1) :
    result,_ = sess.run( [loss,train], feed_dict={x0:mnist.train.next_batch(batch_size)[0],learning_rate:rate})
    if index == 1 or index == batches :
        print "index :",index,", loss:", result



if __name__ == "__main__" :
  print "Start training test ..."
  doEpochOfTraining( loss_6 , train_6 , batches=1 , batch_size=5 )
  print "... finished training test."
  exit()

















import falcon
import json
import random
import os.path
from scipy import spatial

def getImageWithIndex(index) :
    return mnist.test.images[index:index+1]

def getExample(index,layer) :
    return sess.run(layer,feed_dict={x0:getImageWithIndex(index)} ).reshape([SIZE,SIZE])

def arrayToImage(data) :
    import scipy.misc
    import tempfile
    with tempfile.TemporaryFile() as fp :
        scipy.misc.toimage( data ).save( fp=fp, format="PNG" )
        fp.seek(0)
        return fp.read()

def falconRespondArrayAsImage(data,resp) :
    resp.content_type = 'image/png'
    resp.body = arrayToImage(data)


print """
================================
Define the rest endpoints
================================
"""
all_embeddings = []

class Ping:
    def on_get(self, req, resp):
        resp.body = json.dumps( { 'response': 'ping' } )


class Display:
    def on_get(self, req, resp, file_name):
        if not os.path.isfile("view/"+file_name) :
            return

        result = open("view/"+file_name,"r")
        if ( "html" in file_name) :
            resp.content_type = "text/html"
        else :
            resp.content_type = "text/plain"
        
        resp.body = result.read()
        result.close()



class LayerImage:
    def on_get(self, req, resp, layer, index, junk) :
      try :
        ml_layer = [x_noisy,x_out_1,x_out_2,x_out_3,x_out_4,x_out_5,x_in][int(layer)]
        falconRespondArrayAsImage( 
          getExample(int(index),ml_layer) , 
          resp 
          )
      except :
        pass


class BlendImage:
    def on_get(self, req, resp, a_value, b_value, amount) :
      try :
        amount = int(amount) / 100.0
        a_embed = sess.run(conv5e,feed_dict={x0:getImageWithIndex(int(a_value))} )
        b_embed = sess.run(conv5e,feed_dict={x0:getImageWithIndex(int(b_value))} )
        blend_embed = a_embed * amount + b_embed * ( 1 - amount )
        output = sess.run(x_out_5,feed_dict={conv5e:blend_embed,x0:getImageWithIndex(int(a_value))} )
        falconRespondArrayAsImage( output.reshape([SIZE,SIZE]) , resp )
      except :
        pass


class DoLearning:
    def on_get(self, req, resp, index) :
        print "TRAINING WITH",index
        doEpochOfTraining([loss_1,loss_2,loss_3,loss_4,loss_5,loss_6][int(index)],[train_1,train_2,train_3,train_4,train_5,train_6][int(index)])
        global all_embeddings
        all_embeddings = []
        resp.body = json.dumps( { 'response': 'done'} )

class ResetSession:
    def on_get(self, req, resp) :
      resetSession();
      global all_embeddings
      all_embeddings = []
      resp.body = json.dumps( { 'response': 'done'} )




def updateEmbeddings() :
    global all_embeddings
    all_embeddings = sess.run(conv5e,feed_dict={x0:mnist.test.images} ).reshape([-1,SIZE])

def getEmbeddings() :
    global all_embeddings
    if len(all_embeddings) == 0 :
      updateEmbeddings()
    return all_embeddings


class UpdateEmbeddings :
    def on_get(self, req, resp):
      updateEmbeddings()
      resp.body = json.dumps( { 'response': 'done'} )

def calculateDistance(index1,index2) :
    return spatial.distance.cosine( index1, index2 )

def nearestNeighbour(embedding) :
    the_embeddings = getEmbeddings()
    index_list = range(the_embeddings.shape[0])
    distances = np.array([ calculateDistance(embedding,the_embeddings[other]) for other in index_list ])
    nearest = np.argsort( distances )[:10]
    return np.array(index_list)[nearest]

class Similar:
    def on_get(self, req, resp, index):
        the_embeddings = getEmbeddings()
        names = nearestNeighbour( the_embeddings[int(index)] ).tolist()
        resp.body = json.dumps( { 'response' : names } )

class Difference:
    def on_get(self, req, resp, positive, negative):
        the_embeddings = getEmbeddings()
        names = nearestNeighbour( the_embeddings[int(positive)] * 2 - the_embeddings[int(negative)] ).tolist()
        resp.body = json.dumps( { 'response' : names } )



print """
================================
Add the endpoints to the service
================================
"""
api = falcon.API()
api.add_route('/ping', Ping())
api.add_route('/view/{file_name}', Display())
api.add_route('/layer{layer}/{index}/{junk}', LayerImage())
api.add_route('/learn/{index}', DoLearning())
api.add_route('/reset_session', ResetSession())
api.add_route('/update_embeddings', UpdateEmbeddings())
api.add_route('/similar/{index}', Similar())
api.add_route('/difference/{positive}/{negative}', Difference())
api.add_route('/blend/{a_value}/{b_value}/{amount}', BlendImage())

