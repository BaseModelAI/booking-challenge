from sklearn.random_projection import GaussianRandomProjection
from sklearn.preprocessing import StandardScaler
import numpy as np
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)


class Coder(object):
    def __init__(self, n_sketches, sketch_dim):
        self.n_sketches = n_sketches
        self.sketch_dim = sketch_dim
        self.ss = StandardScaler()
        self.sp = GaussianRandomProjection(n_components = 16*n_sketches)

    def fit(self, v):
        self.ss = self.ss.fit(v)
        vv = self.ss.transform(v)
        self.sp = self.sp.fit(vv)
        vvv = self.sp.transform(vv)
        self.init_biases(vvv)

    def transform(self, v):
        v = self.ss.transform(v)
        v = self.sp.transform(v)
        v = self.discretize(v)
        v = np.packbits(v, axis=-1)
        v = np.frombuffer(np.ascontiguousarray(v), dtype=np.uint16).reshape(v.shape[0], -1) % self.sketch_dim
        return v

    def transform_to_absolute_codes(self, v, labels=None):
        codes = self.transform(v)
        pos_index = np.array([i*self.sketch_dim for i in range(self.n_sketches)], dtype=np.int_)
        index = codes + pos_index
        return index


class DLSH(Coder):
    def __init__(self, n_sketches, sketch_dim):
        super().__init__(n_sketches, sketch_dim)

    def init_biases(self, v):
        self.biases = np.array([np.percentile(v[:, i], q=50, axis=0) for i in range(v.shape[1])])

    def discretize(self, v):
        return ((np.sign(v - self.biases)+1)/2).astype(np.uint8)
