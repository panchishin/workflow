print(__doc__)
from time import time

import numpy as np
import session
session.restoreSession()
import embeddings
import matplotlib.pyplot as plt
from matplotlib import offsetbox
from sklearn import (manifold, datasets, decomposition, ensemble,
                     discriminant_analysis, random_projection)

def get_mnist_data() :
  from tensorflow.examples.tutorials.mnist import input_data
  return input_data.read_data_sets('./cache', one_hot=True)

mnist = get_mnist_data()
embeddings.data_set = mnist.test.images


#----------------------------------------------------------------------
# Scale and visualize the embedding vectors
def plot_embedding(X, y, title=None):
    x_min, x_max = np.min(X, 0), np.max(X, 0)
    X = (X - x_min) / (x_max - x_min)
    plt.figure()
    ax = plt.subplot(111)
    for i in range(X.shape[0]):
        plt.text(X[i, 0], X[i, 1], str(y[i]),
                 color=plt.cm.Set1(y[i] / 10.),
                 fontdict={'weight': 'bold', 'size': 9})
    if hasattr(offsetbox, 'AnnotationBbox'):
        # only print thumbnails with matplotlib > 1.0
        shown_images = np.array([[1., 1.]])  # just something big
        for i in range(X.shape[0]):
            dist = np.sum((X[i] - shown_images) ** 2, 1)
            if np.min(dist) < 4e-3:
                # don't show points that are too close
                continue
            shown_images = np.r_[shown_images, [X[i]]]
            imagebox = offsetbox.AnnotationBbox(
                offsetbox.OffsetImage(mnist.test.images[i].reshape([28,28])[::2,::2], cmap=plt.cm.gray_r),
                X[i])
            ax.add_artist(imagebox)
    plt.xticks([]), plt.yticks([])
    if title is not None:
        plt.title(title)


#----------------------------------------------------------------------
# t-SNE embedding of the digits dataset
tsne = manifold.TSNE(n_components=2, init='pca', random_state=0)
t0 = time()

X = mnist.test.images[:5000,:]
y = np.argmax(mnist.test.labels[:5000,:],1)
X_tsne = tsne.fit_transform(X)
plot_embedding(X_tsne,y,
               "t-SNE images of the digits (time %.2fs)" %
               (time() - t0))

t0 = time()
X = embeddings.getEmbeddings()[:5000,:]
X_tsne = tsne.fit_transform(X)
plot_embedding(X_tsne,y,
               "t-SNE embedding of the digits (time %.2fs)" %
               (time() - t0))
plt.show()
