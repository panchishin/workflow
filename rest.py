import session
session.restoreSession()
import falcon
import os.path  # for serving html files
import json
import random
import numpy as np
from scipy import spatial
autoencode_model = session.autoencode_model
from embeddings import Embeddings
embeddings = Embeddings()
import label_predict
from sklearn.manifold import TSNE

from data_source import LazyLoadWrapper, BatchWrapper, ResizeWrapper, ReshapeWrapper, Mnist
imageData = LazyLoadWrapper( BatchWrapper( ResizeWrapper( ReshapeWrapper( Mnist(), [28,28,1] ) , [32,32] ) ) )

embeddings.data_set = imageData.getImages()


def getImageWithIndex(index) :
  return imageData.getImages()[index:index+1]

def getExample(index,layer) :
  return session.sess.run(layer,feed_dict={autoencode_model.x_in:getImageWithIndex(index)} ).reshape([autoencode_model.SIZE,autoencode_model.SIZE])

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
    except Exception as e:
      print(e)


class BlendImage:
  def on_get(self, req, resp, a_value, b_value, amount) :
    try :
      amount = int(amount) / 100.0
      a_embed = session.sess.run(autoencode_model.embedding,feed_dict={autoencode_model.x_in:getImageWithIndex(int(a_value))} )
      b_embed = session.sess.run(autoencode_model.embedding,feed_dict={autoencode_model.x_in:getImageWithIndex(int(b_value))} )
      blend_embed = a_embed * amount + b_embed * ( 1 - amount )
      output = session.sess.run(autoencode_model.x_out_5,feed_dict={autoencode_model.embedding:blend_embed,autoencode_model.x_in:getImageWithIndex(int(a_value))} )
      falconRespondArrayAsImage( output.reshape([autoencode_model.SIZE,autoencode_model.SIZE]) , resp )
    except Exception as e:
      print(e)


class DoLearning:
  def on_get(self, req, resp, index) :
    print "TRAINING WITH",index
    session.doEpochOfTraining(
      [autoencode_model.loss_1,autoencode_model.loss_2,autoencode_model.loss_3,autoencode_model.loss_4,autoencode_model.loss_5,autoencode_model.loss_6][int(index)],
      [autoencode_model.train_1,autoencode_model.train_2,autoencode_model.train_3,autoencode_model.train_4,autoencode_model.train_5,autoencode_model.train_6][int(index)],
      imageData)
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




class GroupPredict:
  def __init__(self) :
    self.previous_group_predict_result = None
    self.previous_group_predict_data_hash = 0
    self.previous_group_predict_page = 0
    self.previous_response_index = -1

  def report(self, result) :
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
    ground = np.argmax( imageData.getLabels() , 1 )
    for a in range(10) :
      precision = 1. * (( ground == a ) * ( predict == a )).sum() / ( predict == a ).sum()
      recall = 1. * (( ground == a ) * ( predict == a )).sum() / ( ground == a ).sum()
      f1 = ( 2. * precision * recall / ( precision + recall + 0.01 ))
      total_f1 += f1
      print "%5d" % ( 100. * f1 ),
    print "AVG %5d" % ( 10. * total_f1 )
    print "== Ground truth error %5.2f == error,top_percent,count : " % ( 100.* (np.argmax( imageData.getLabels() , 1 ) != np.argmax( result , 1 ) ).mean() ),
    for confidence in [1.,.99,.90,.75,.5,.25,.1,.01] :
      conf_filter = np.argsort( np.max( result , 1 ) )[::-1]
      conf_filter = conf_filter[ : int( conf_filter.shape[0] * confidence ) ]
      print "%5.2f%s %4.2f %5d ," % ( 100.* (np.argmax( imageData.getLabels()[conf_filter,:] , 1 ) != np.argmax( result[conf_filter,:] , 1 ) ).mean(), "%", confidence, conf_filter.shape[0] ),
    print "\n"

  def on_post(self, req, resp, response_index):
    response_index = int(response_index)
    
    data_text = req.stream.read()
    data = json.loads( data_text )
    grouping = data['grouping']

    if hash( json.dumps(grouping) ) == self.previous_group_predict_data_hash :
      result = self.previous_group_predict_result
      if response_index == self.previous_response_index :
        self.previous_group_predict_page = (self.previous_group_predict_page + 1) % 5
      else :
        self.previous_response_index = response_index
        self.previous_group_predict_page = 0
    else :
      result,error = label_predict.predictiveMultiClassWeights(grouping,embeddings)
      self.previous_group_predict_result = result
      self.previous_group_predict_page = 0
      self.previous_group_predict_data_hash = hash( json.dumps(grouping) )
      self.previous_error = error
      if (result.shape[1] == 10) :
        self.report(result)

    max_result = np.max(result,1)
    if response_index >= 0 :
      result = np.array(result)[:,response_index]
    else :
      result = max_result

    isLabeled = np.zeros( result.shape[0] )
    for category in range(len(grouping)) :
      if category == response_index :
        for example in grouping[category] :
          isLabeled[example] = 1
    
    if data['isLabeled'] == 1 :
      likely_filter = ( isLabeled >= 1. )
    else :
      likely_filter = ( isLabeled < 1. ) * ( result == max_result )

    likely_weight = result[ likely_filter ]
    likely_index  = np.array(range(result.shape[0]))[ likely_filter ]
    positive = likely_index[np.argsort(likely_weight)]

    if data['order'] == 'forward' :
      positive = positive[::-1]

    index = int( positive.shape[0] * data['index'] )
    positive = positive[index:]

    positive = positive[self.previous_group_predict_page*10:][:10].tolist()

    resp.body = json.dumps( { 'response' : { 'positive' : positive , 'error' : self.previous_error.tolist() } } )



class TSne:

  def calculate_tsne(self):
    print "Starting the TSNE calculation ..."
    self.max_size = 2500
    positions = TSNE(n_components=2).fit_transform(embeddings.getEmbeddings()[:self.max_size])
    positions = positions - np.min( positions , 0 )
    positions = positions / np.max( positions , 0 )
    self.positions = positions
    print "... finished the TSNE calculations."

  def __init__(self) :
    self.primed = False

  def on_get(self, req, resp, size):
    if self.primed == False :
      self.primed = True
      self.calculate_tsne()
    size = min( int(size) , self.max_size)
    data = []
    for index in range(size) :
      data.append( { 'x' : self.positions[index,0] , 'y' : self.positions[index,1] , 'id' : index } )

    resp.body = json.dumps( { 'response' : data } )



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
api.add_route('/tsne/{size}', TSne())

