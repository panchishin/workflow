import falcon
import os.path  # for serving html files
import json
import random
import numpy as np
from scipy import spatial
import model


model.restoreSession()

def get_mnist_data() :
  from tensorflow.examples.tutorials.mnist import input_data
  return input_data.read_data_sets('./cache', one_hot=True)

mnist = get_mnist_data()



def getImageWithIndex(index) :
    return mnist.test.images[index:index+1]

def getExample(index,layer) :
    return model.sess.run(layer,feed_dict={model.x0:getImageWithIndex(index)} ).reshape([model.SIZE,model.SIZE])

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
        ml_layer = [model.x_noisy,model.x_out_1,model.x_out_2,model.x_out_3,model.x_out_4,model.x_out_5,model.x_in][int(layer)]
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
        a_embed = model.sess.run(model.conv5e,feed_dict={model.x0:getImageWithIndex(int(a_value))} )
        b_embed = model.sess.run(model.conv5e,feed_dict={model.x0:getImageWithIndex(int(b_value))} )
        blend_embed = a_embed * amount + b_embed * ( 1 - amount )
        output = model.sess.run(model.x_out_5,feed_dict={model.conv5e:blend_embed,model.x0:getImageWithIndex(int(a_value))} )
        falconRespondArrayAsImage( output.reshape([model.SIZE,model.SIZE]) , resp )
      except :
        pass


class DoLearning:
    def on_get(self, req, resp, index) :
        print "TRAINING WITH",index
        model.doEpochOfTraining(
          [model.loss_1,model.loss_2,model.loss_3,model.loss_4,model.loss_5,model.loss_6][int(index)],
          [model.train_1,model.train_2,model.train_3,model.train_4,model.train_5,model.train_6][int(index)],
          mnist.train)
        global all_embeddings
        all_embeddings = []
        resp.body = json.dumps( { 'response': 'done'} )

class ResetSession:
    def on_get(self, req, resp) :
      model.resetSession();
      global all_embeddings
      all_embeddings = []
      resp.body = json.dumps( { 'response': 'done'} )

class RestoreSession:
    def on_get(self, req, resp) :
      model.restoreSession();
      resp.body = json.dumps( { 'response': 'done'} )

class SaveSession:
    def on_get(self, req, resp) :
      model.saveSession();
      resp.body = json.dumps( { 'response': 'done'} )


def updateEmbeddings() :
    global all_embeddings
    all_embeddings = model.sess.run(model.conv5e,feed_dict={model.x0:mnist.test.images} ).reshape([-1,model.SIZE])


def getEmbeddings() :
    global all_embeddings
    if len(all_embeddings) == 0 :
      updateEmbeddings()
    return all_embeddings

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
api.add_route('/save_session', SaveSession())
api.add_route('/restore_session', RestoreSession())
api.add_route('/similar/{index}', Similar())
api.add_route('/difference/{positive}/{negative}', Difference())
api.add_route('/blend/{a_value}/{b_value}/{amount}', BlendImage())

