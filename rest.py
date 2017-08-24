import session
session.restoreSession()
import falcon
import os.path  # for serving html files
import json
import random
import numpy as np
from scipy import spatial
import autoencode_model
import embeddings
import label_predict


def get_mnist_data() :
  from tensorflow.examples.tutorials.mnist import input_data
  return input_data.read_data_sets('./cache', one_hot=True)

mnist = get_mnist_data()
embeddings.data_set = mnist.train.images


def getImageWithIndex(index) :
  return mnist.train.images[index:index+1]

def getExample(index,layer) :
  return session.sess.run(layer,feed_dict={autoencode_model.x0:getImageWithIndex(index)} ).reshape([autoencode_model.SIZE,autoencode_model.SIZE])

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


"""
================================
Define the rest endpoints
================================
"""

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
      ml_layer = [autoencode_model.x_noisy,autoencode_model.x_out_1,autoencode_model.x_out_2,autoencode_model.x_out_3,autoencode_model.x_out_4,autoencode_model.x_out_5,autoencode_model.x_in][int(layer)]
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
      a_embed = session.sess.run(autoencode_model.conv5e,feed_dict={autoencode_model.x0:getImageWithIndex(int(a_value))} )
      b_embed = session.sess.run(autoencode_model.conv5e,feed_dict={autoencode_model.x0:getImageWithIndex(int(b_value))} )
      blend_embed = a_embed * amount + b_embed * ( 1 - amount )
      output = session.sess.run(autoencode_model.x_out_5,feed_dict={autoencode_model.conv5e:blend_embed,autoencode_model.x0:getImageWithIndex(int(a_value))} )
      falconRespondArrayAsImage( output.reshape([autoencode_model.SIZE,autoencode_model.SIZE]) , resp )
    except :
      pass


class DoLearning:
  def on_get(self, req, resp, index) :
    print "TRAINING WITH",index
    session.doEpochOfTraining(
      [autoencode_model.loss_1,autoencode_model.loss_2,autoencode_model.loss_3,autoencode_model.loss_4,autoencode_model.loss_5,autoencode_model.loss_6][int(index)],
      [autoencode_model.train_1,autoencode_model.train_2,autoencode_model.train_3,autoencode_model.train_4,autoencode_model.train_5,autoencode_model.train_6][int(index)],
      mnist.train)
    embeddings.reset()
    resp.body = json.dumps( { 'response': 'done'} )

class ResetSession:
  def on_get(self, req, resp) :
    session.resetSession();
    embeddings.reset()
    resp.body = json.dumps( { 'response': 'done'} )

class RestoreSession:
  def on_get(self, req, resp) :
    session.restoreSession();
    resp.body = json.dumps( { 'response': 'done'} )

class SaveSession:
  def on_get(self, req, resp) :
    session.saveSession();
    resp.body = json.dumps( { 'response': 'done'} )


class Similar:
  def on_get(self, req, resp, index):
    names = embeddings.nearestNeighbourByIndex( int(index) ).tolist()
    resp.body = json.dumps( { 'response' : names } )


previous_group_predict_result = None
previous_group_predict_data_hash = 0
previous_group_predict_page = 0

class GroupPredict:
  def on_post(self, req, resp, response_index):
    response_index = int(response_index)
    
    global previous_group_predict_result , previous_group_predict_data_hash , previous_group_predict_page
    data_text = req.stream.read()
    data = json.loads( data_text )

    if hash( json.dumps(data['grouping']) ) == previous_group_predict_data_hash :
      result = previous_group_predict_result
      previous_group_predict_page = (previous_group_predict_page + 1) % 20
    else :
      result = label_predict.predictiveMultiClassWeights(data['grouping'],embeddings)
      previous_group_predict_result = result
      previous_group_predict_page = 0
      previous_group_predict_data_hash = hash( json.dumps(data['grouping']) )
      if (result.shape[1] == 10) :

        print "class :",
        for b in range(10) :
          print "%5d" % b,
        print " "

        print "count :",
        predict = np.argmax( result , 1 )
        prediction_sums = [ ( predict == b ).sum() for b in range(10) ]
        for item in prediction_sums :
          print "%5d" % item,
        rms = (sum( [ (item - sum(prediction_sums)/10.)**2 for item in prediction_sums ] )/10.)**.5
        print "RMS %5d" % rms

        print "  F1  :",
        total_f1 = 0.
        ground = np.argmax( mnist.train.labels , 1 )
        for a in range(10) :
          precision = 1. * (( ground == a ) * ( predict == a )).sum() / ( predict == a ).sum()
          recall = 1. * (( ground == a ) * ( predict == a )).sum() / ( ground == a ).sum()
          f1 = ( 2. * precision * recall / ( precision + recall + 0.01 ))
          total_f1 += f1
          print "%5d" % ( 100. * f1 ),
        print "AVG %5d" % ( 10. * total_f1 )
        print "== Ground truth error %5.2f ==\n" % ( 100.* (np.argmax( mnist.train.labels , 1 ) != np.argmax( result , 1 ) ).mean() )

    max_result = np.max(result,1)
    if response_index >= 0 :
      result = np.array(result)[:,int(response_index)]
    else :
      result = max_result
    
    likely_filter = ( result < 1. ) * ( result > .0 ) * ( result == max_result )
    likely_weight = result[ likely_filter ]
    likely_index  = np.array(range(result.shape[0]))[ likely_filter ]
    positive = likely_index[np.argsort(likely_weight)]

    if data['confidence'] == "low" :
      positive = positive[previous_group_predict_page*20:][:10].tolist()
    elif data['confidence'] == "medium" :
      middle = positive.shape[0] / 2 + (previous_group_predict_page - 10)*20
      positive = positive[middle:][:10].tolist()
    else :
      positive = positive[::-1][previous_group_predict_page*20:][:10].tolist()

    resp.body = json.dumps( { 'response' : { 'positive' : positive } } )



"""
================================
Add the endpoints to the service
================================
"""
api = falcon.API()
api.add_route('/view/{file_name}', Display())
api.add_route('/layer{layer}/{index}/{junk}', LayerImage())
api.add_route('/learn/{index}', DoLearning())
api.add_route('/reset_session', ResetSession())
api.add_route('/save_session', SaveSession())
api.add_route('/restore_session', RestoreSession())
api.add_route('/similar/{index}', Similar())
api.add_route('/blend/{a_value}/{b_value}/{amount}', BlendImage())
api.add_route('/group_predict/{response_index}', GroupPredict())


