import numpy as np
import label_predict
from embeddings import Embeddings
import nearest_neighbour
from data_source import LazyLoadWrapper, ResizeWrapper, ReshapeWrapper, Mnist

embeddings = Embeddings()

imageData = LazyLoadWrapper(ResizeWrapper(ReshapeWrapper(Mnist(False), [28, 28, 1]), [32, 32]))

embeddings.data_set = imageData.getImages()

number = 4

print "The number is", np.argmax(imageData.getLabels()[[number]], 1)

nearest = nearest_neighbour.byIndex(number, embeddings.getEmbeddings(), size=200)
result = zip(imageData.getLabels()[nearest], nearest)
nearest = []
negative_examples = []
for label, data_index in result:
    label = np.argmax(label)
    if label == 4:
        nearest.append(data_index)
    else:
        negative_examples.append(data_index)

print "Pretend labeling the first", len(nearest), " ..."

fours_found = 0
fours_missed = 0
result = label_predict.predictiveMultiClassWeights([nearest, negative_examples], embeddings)[0]
for item in zip(np.argmax(imageData.getLabels()[:100, :], 1), result[:100]):
    if item[0]*1 == 4:
        if np.argmax(item[1]) == 0 :
            fours_found += 1
        else :
            fours_missed += 1
    else :
        if np.argmax(item[1]) == 0 :
            fours_missed += 1
    # print item[0], ["is a '4'", "not", "something else"][np.argmax(item[1])]

if fours_found > fours_missed and fours_found > 4:
	print "done"
else :
	print "FAILURE.  Not enough 4's found!"