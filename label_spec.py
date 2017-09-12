import session
session.restoreSession()
import numpy as np
import embeddings
import label_predict

from data_source import LazyLoadWrapper, BatchWrapper, ResizeWrapper, ReshapeWrapper, Mnist
imageData = LazyLoadWrapper( ResizeWrapper( ReshapeWrapper( Mnist(False), [28,28,1] ) , [32,32] ) )

embeddings.data_set = imageData.getImages()

number = 4

print "The number is",np.argmax( imageData.getLabels()[ [number] ] , 1 )
print "And its index is",number

nearest = embeddings.nearestNeighbourByIndex(number,200)
result = zip( imageData.getLabels()[ nearest ] , nearest )
nearest = []
negative_examples = []
for label,data_index in result :
  label = np.argmax(label)
  if label == 4 :
    nearest.append(data_index)
  else :
    negative_examples.append(data_index)

print "Pretend labeling the first",len(nearest)," ..."


result = label_predict.predictiveMultiClassWeights( [ nearest,negative_examples] ,embeddings)[0]
for item in zip( np.argmax(imageData.getLabels()[:20,:],1), result[:20] ) :
  print item[0], ["is a '4'","not","something else"][ np.argmax(item[1]) ]
