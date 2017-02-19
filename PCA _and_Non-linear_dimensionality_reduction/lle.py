
# -*- coding: utf-8 -*-
from scipy.spatial.distance import cdist


"""
===================================
Swiss Roll reduction with LLE
===================================
An illustration of Swiss Roll reduction
with locally linear embedding
"""

################################################################################
# locally linear embedding function

from scipy.sparse import linalg, eye
from sklearn import neighbors
from scipy.sparse.linalg import eigsh

def locally_linear_embedding(X, n_neighbors, out_dim, tol=1e-6, max_iter=200):
    W = neighbors.kneighbors_graph(
       X, n_neighbors=n_neighbors, mode='distance')

    A = eye(*W.shape, format=W.format) - W
    A = (A.T).dot(A).tocsr()
    eigen_values, eigen_vectors = eigsh(A,k= 2)

    return eigen_vectors, np.sum(eigen_values)


import numpy as np
import pylab as pl

################################################################################
# generate the swiss roll

n_samples, n_features = 2000, 3
n_turns, radius = 1.2, 1.0
rng = np.random.RandomState(0)
t = rng.uniform(low=0, high=1, size=n_samples)
data = np.zeros((n_samples, n_features))

# generate the 2D spiral data driven by a 1d parameter t
max_rot = n_turns * 2 * np.pi
data[:, 0] = radius = t * np.cos(t * max_rot)
data[:, 1] = radius = t * np.sin(t * max_rot)
data[:, 2] = rng.uniform(-1, 1.0, n_samples)
manifold = np.vstack((t * 2 - 1, data[:, 2])).T.copy()
colors = manifold[:, 0]

# rotate and plot original data
sp = pl.subplot(211)
U = np.dot(data, [[-.79, -.59, -.13],
                  [ .29, -.57,  .75],
                  [-.53,  .56,  .63]])
sp.scatter(U[:, 1], U[:, 2], c=colors)
sp.set_title("Original data")


print "Computing LLE embedding"
n_neighbors, out_dim = 12, 2
X_r, cost = locally_linear_embedding(data, n_neighbors, out_dim)

sp = pl.subplot(212)
sp.scatter(X_r[:,0], X_r[:,1], c=colors)
sp.set_title("LLE embedding")
pl.show()