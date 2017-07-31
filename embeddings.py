import numpy as np
import autoencode_model
import session
from scipy import spatial

data_set = []

all_embeddings = []
def reset() :
    global all_embeddings
    all_embeddings = []

def update() :
    global all_embeddings
    all_embeddings = session.sess.run(autoencode_model.conv5e,feed_dict={autoencode_model.x0:data_set} ).reshape([-1,autoencode_model.SIZE])

def getEmbeddings() :
    global all_embeddings
    if len(all_embeddings) == 0 :
      update()
    return all_embeddings

def calculateDistance(index1,index2) :
    return spatial.distance.cosine( index1, index2 )

def nearestNeighbour(embedding,size=10) :
    the_embeddings = getEmbeddings()
    index_list = range(the_embeddings.shape[0])
    distances = np.array([ calculateDistance(embedding,the_embeddings[other]) for other in index_list ])
    nearest = np.argsort( distances )[:size]
    return np.array(index_list)[nearest]

def nearestNeighbourByIndex(index,size=10) :
    the_embeddings = getEmbeddings()
    embedding = the_embeddings[index]
    index_list = range(the_embeddings.shape[0])
    distances = np.array([ calculateDistance(embedding,the_embeddings[other]) for other in index_list ])
    nearest = np.argsort( distances )[:size]
    return np.array(index_list)[nearest]
