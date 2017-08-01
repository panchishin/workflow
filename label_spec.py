import session
session.restoreSession()
import numpy as np
import embeddings
import label_predict

def get_mnist_data() :
  from tensorflow.examples.tutorials.mnist import input_data
  return input_data.read_data_sets('./cache', one_hot=True)

mnist = get_mnist_data()
embeddings.data_set = mnist.test.images

number = 4

print "The number is",np.argmax( mnist.test.labels[ [number] ] , 1 )
print "And its index is",number

nearest = embeddings.nearestNeighbourByIndex(number,200)
result = zip( mnist.test.labels[ nearest ] , nearest )
nearest = []
negative_examples = []
for label,data_index in result :
  label = np.argmax(label)
  if label == 4 :
    nearest.append(data_index)
  else :
    negative_examples.append(data_index)

print "Pretend labeling the first",len(nearest)," ..."

result = label_predict.predictiveBinaryWeights(nearest,negative_examples,embeddings)
for item in zip( np.argmax(mnist.test.labels[:20,:],1), result[:20] ) :
  print item

result = label_predict.predictiveMultiClassWeights( [ nearest,negative_examples] ,embeddings)
for item in zip( np.argmax(mnist.test.labels[:20,:],1), result[:20] ) :
  print item
