import numpy as np
from scipy.sparse import linalg,eye
from sklearn import neighbors


from scipy.sparse import spdiags, issparse
from scipy.sparse.linalg import lobpcg, LinearOperator

def new_relax(A,x,b):
 x[:] += 0.125*(b - A*x)
A = gallery.poisson( (100,100), format='csr')
b = ones( (A.shape[0],1))
res = []
ml = smoothed_aggregation_solver(A)
ml.levels[0].presmoother = new_relax
ml.levels[0].postsmoother = new_relax
x = ml.solve(b, tol=1e-8, residuals=res)
semilogy(res[1:])
show()