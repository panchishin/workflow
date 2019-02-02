import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

execfile("rest.py")


dataset.imageData = dataset.choose_mnist()
dataset.value = 'mnist'
dataset.dataSize = dataset.imageData.getImages().shape
print "The embeddings are of shape",embeddings.getEmbeddings().shape


f,a=plt.subplots(4,8,figsize=(32,4))
for i in range(32):
    _ = a[int(i/8)][i%8].hist(embeddings.getEmbeddings()[:,i],20)

plt.show()


import pandas as pd
dataframe = pd.DataFrame(embeddings.getEmbeddings())
corr = plt.matshow(dataframe.corr().abs(), cmap='gray')
plt.show()


"""
count = 40

image = dataset.imageData.getImages()[0:count]
layer = autoencode_model.x_out_5

# recreate from image
data = predictor.sess.run(layer, feed_dict={autoencode_model.x_in: image})

# recreate embedding
data = predictor.sess.run(layer, feed_dict={autoencode_model.embedding: embeddings.getEmbeddings()[:count].reshape([count,1,1,32])})

f,a=plt.subplots(count/5,5,figsize=(32,4))
for i in range(count):
    a[i/5][i%5].imshow(np.reshape(data[i],(32,32)))

plt.show()


# create a morphing from '4' to '6'
four = embeddings.getEmbeddings()[2:3].reshape([1,1,32])
six = embeddings.getEmbeddings()[3:4].reshape([1,1,32])
eight = embeddings.getEmbeddings()[5:6].reshape([1,1,32])
zero = embeddings.getEmbeddings()[7:8].reshape([1,1,32])
nine = embeddings.getEmbeddings()[8:9].reshape([1,1,32])
seven0 = embeddings.getEmbeddings()[0:1].reshape([1,1,32])
seven35 = embeddings.getEmbeddings()[35:36].reshape([1,1,32])

items = [four,six,eight,zero,nine,seven0,seven35]
items = zip(items[:-1],items[1:])

for pair in items :
    morph = np.zeros([10,1,1,32])
    for i in range(10):
        morph[i] = pair[0]*(9-i)/9.0 + pair[1]*(i)/9.0
    # recreate embedding
    data = predictor.sess.run(layer, feed_dict={autoencode_model.embedding: morph})
    f,a=plt.subplots(2,5) #,figsize=(32,4))
    for i in range(10):
        _ = a[int(i/5)][i%5].imshow(np.reshape(data[i],(32,32)))
    plt.show()



# random
randoms = np.random.random([10,1,1,32]) - 0.5
randoms = randoms * np.abs(randoms)

randoms = np.zeros([10,32])
for i in range(10) :
    for x in range(3) :
        randoms[i][np.random.randint(32)] = np.random.random() * 0.5
        randoms[i][np.random.randint(32)] = -1 * np.random.random() * 0.5

# recreate embedding
data = predictor.sess.run(layer, feed_dict={autoencode_model.embedding: randoms.reshape([10,1,1,32])})

f,a=plt.subplots(2,5,figsize=(32,4))
for i in range(10):
    a[int(i/5)][i%5].imshow(np.reshape(data[i],(32,32)))

plt.show()




def calc(sparsity,embed) :
    return \
          sparsity * np.log(sparsity) \
        - sparsity * np.log(embed) \
        + (1 - sparsity) * np.log(1. - sparsity) \
        - (1 - sparsity) * np.log(1. - embed)

calc(.99,np.array([.2,.3,.4,.5]))

"""