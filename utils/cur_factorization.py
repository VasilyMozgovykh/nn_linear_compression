import matrix_fact
import numpy as np
import scipy.sparse

from matrix_fact.svd import pinv
from matrix_fact.base import MatrixFactBase3

__all__ = ['CUR']

class CUR(MatrixFactBase3):

    def __init__(self, data, k=-1, rrank=0, crank=0):
        MatrixFactBase3.__init__(self, data,k=k,rrank=rrank, crank=rrank)

        self._rset = range(self._rows)
        self._cset = range(self._cols)

    def sample(self, s, probs):

        prob_rows = np.cumsum(probs.flatten())
        temp_ind = np.zeros(s, np.int32)

        for i in range(s):
            v = np.random.rand()

            try:
                tempI = np.where(prob_rows >= v)[0]
                temp_ind[i] = tempI[0]
            except:
                temp_ind[i] = len(prob_rows)

        return np.sort(temp_ind)

    def sample_probability(self):

        if scipy.sparse.issparse(self.data):
            dsquare = self.data.multiply(self.data)
        else:
            dsquare = self.data[:,:]**2

        pcol = np.array(dsquare.sum(axis=0), np.float64)

        pcol /= pcol.sum()

        return pcol.reshape(-1, 1)
    def computeUCR(self):

        if scipy.sparse.issparse(self.data):
            self._C = self.data[:, self._cid] * scipy.sparse.csc_matrix(np.diag(self._ccnt**(1/2)))
            self._U = pinv(self._C, self._k) * self.data[:,:]# * pinv(self._R, self._k)

        else:
            self._C = np.dot(self.data[:, self._cid].reshape((self._rows, -1)), np.diag(self._ccnt**(1/2)))

            self._U = np.dot(pinv(self._C, self._k), self.data[:,:])

        # set some standard (with respect to SVD) variable names 
        self.U = self._C
        self.S = self._U

    def factorize(self):

        pcol = self.sample_probability()
        self._cid = self.sample(self._crank, pcol)
        self._ccnt = np.ones(len(self._cid))

        self.computeUCR()