import numpy as np

from models.recommend import RecommenderMixin


class ImplicitItemKNN(RecommenderMixin):
    def __init__(self, k=50):
        self.k = k

    def fit(self, P):
        self.P = P.tocsr()
        sim = (self.P.T @ self.P).toarray()
        norms = np.sqrt(np.diag(sim))
        sim = sim / (np.outer(norms, norms) + 1e-9)
        np.fill_diagonal(sim, 0)

        weak = np.argsort(sim, axis=1)[:, : -self.k]
        np.put_along_axis(sim, weak, 0, axis=1)
        self.sim = sim
        return self

    def score(self, users):
        return self.P[users] @ self.sim
