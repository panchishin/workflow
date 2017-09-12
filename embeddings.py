import numpy as np
import random
from scipy import spatial

class Embeddings:
  def __init__(self, session=None, model=None, max_search_size=2000) :
    self.data_set = []
    self.all_embeddings = []
    self.max_search_size = max_search_size
    if session == None :
      import session as importSession
      self.session = importSession
    else :
      self.session = session
    if model == None :
      self.model = self.session.autoencode_model
    else :
      self.model = model

  def reset(self) :
    self.all_embeddings = []

  def getEmbeddings(self) :
    if len(self.all_embeddings) == 0 :
      self.all_embeddings = self.session.sess.run(self.model.embedding,
        feed_dict={self.model.x_in:self.data_set} ).reshape([-1,self.model.SIZE])
    return self.all_embeddings
  
  def nearestNeighbourByIndex(self,index,size=10) :
    the_embeddings = self.getEmbeddings()
    embedding = the_embeddings[index]
    offset = random.randint(0,the_embeddings.shape[0]-self.max_search_size-1)
    the_embeddings = the_embeddings[offset:(offset+self.max_search_size),:]
    index_list = range(the_embeddings.shape[0])
    distances = np.array([ spatial.distance.cosine(embedding,the_embeddings[other]) for other in index_list ])
    nearest = np.argsort( distances )[:size]
    return np.array(index_list)[nearest]+offset
