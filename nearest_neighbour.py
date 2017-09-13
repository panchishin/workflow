import numpy as np
import random
from scipy import spatial

def byIndex(target_index,embeddings,size=10,max_search_size=2000) :
  target_embedding = embeddings[target_index]

  # only search a small region
  offset = random.randint(0,embeddings.shape[0]-max_search_size-1)
  embeddings = embeddings[offset:(offset+max_search_size),:]

  index_list = range(embeddings.shape[0])
  distances = np.array([ spatial.distance.cosine(target_embedding,embeddings[other]) for other in index_list ])
  nearest = np.argsort( distances )[:size]

  # nearest is zero'd according to offset.
  # filter nearest and add offset
  return np.array(index_list)[nearest]+offset
