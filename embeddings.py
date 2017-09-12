import numpy as np
import random
import session
from scipy import spatial

class Embeddings:
  def __init__(self) :
    self.data_set = []
    self.all_embeddings = []
    self.autoencode_model = session.autoencode_model

  def reset(self) :
    self.all_embeddings = []
  
  def update(self) :
    self.all_embeddings = session.sess.run(self.autoencode_model.embedding,
        feed_dict={self.autoencode_model.x_in:self.data_set} ).reshape([-1,self.autoencode_model.SIZE])
  
  def getEmbeddings(self) :
    if len(self.all_embeddings) == 0 :
      self.update()
    return self.all_embeddings
  
  def calculateDistance(self,index1,index2) :
    return spatial.distance.cosine( index1, index2 )
  
  def nearestNeighbour(self,embedding,size=10) :
    the_embeddings = self.getEmbeddings()
    index_list = range(the_embeddings.shape[0])
    distances = np.array([ self.calculateDistance(embedding,the_embeddings[other]) for other in index_list ])
    nearest = np.argsort( distances )[:size]
    return np.array(index_list)[nearest]
  
  def nearestNeighbourByIndex(self,index,size=10) :
    max_search_size = 2000
    the_embeddings = self.getEmbeddings()
    embedding = the_embeddings[index]
    offset = random.randint(0,the_embeddings.shape[0]-max_search_size-1)
    the_embeddings = the_embeddings[offset:(offset+max_search_size),:]
    index_list = range(the_embeddings.shape[0])
    distances = np.array([ self.calculateDistance(embedding,the_embeddings[other]) for other in index_list ])
    nearest = np.argsort( distances )[:size]
    return np.array(index_list)[nearest]+offset
