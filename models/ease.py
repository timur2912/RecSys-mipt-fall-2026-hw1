import numpy as np

from models.recommend import RecommenderMixin


class EASE(RecommenderMixin):
    def __init__(self, l2=500.0):
        self.l2 = l2

    def fit(self, P):
        self.P = P.tocsr()
        G = (self.P.T @ self.P).toarray() + self.l2 * np.eye(P.shape[1])
        G_inv = np.linalg.inv(G)
        self.B = -G_inv / np.diag(G_inv)
        np.fill_diagonal(self.B, 0)
        return self

    def score(self, users):
        return self.P[users] @ self.B
