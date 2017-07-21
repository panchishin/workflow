import numpy as np
import model
import session
from scipy import spatial

data_set = []

all_embeddings = []
def reset() :
    global all_embeddings
    all_embeddings = []

def update() :
    global all_embeddings
    all_embeddings = session.sess.run(model.conv5e,feed_dict={model.x0:data_set} ).reshape([-1,model.SIZE])

def getEmbeddings() :
    global all_embeddings
    if len(all_embeddings) == 0 :
      update()
    return all_embeddings

def calculateDistance(index1,index2) :
    return spatial.distance.cosine( index1, index2 )

def nearestNeighbour(embedding) :
    the_embeddings = getEmbeddings()
    index_list = range(the_embeddings.shape[0])
    distances = np.array([ calculateDistance(embedding,the_embeddings[other]) for other in index_list ])
    nearest = np.argsort( distances )[:10]
    return np.array(index_list)[nearest]

def nearestNeighbourByIndex(index) :
    the_embeddings = getEmbeddings()
    embedding = the_embeddings[index]
    index_list = range(the_embeddings.shape[0])
    distances = np.array([ calculateDistance(embedding,the_embeddings[other]) for other in index_list ])
    nearest = np.argsort( distances )[:10]
    return np.array(index_list)[nearest]
