import numpy as np

from models.recommend import RecommenderMixin


class ImplicitALS(RecommenderMixin):
    def __init__(self, n_factors=32, regularization=0.1, alpha=40.0, n_iterations=5, random_state=42):
        self.n_factors = n_factors
        self.regularization = regularization
        self.alpha = alpha
        self.n_iterations = n_iterations
        self.random_state = random_state

    def fit(self, P):
        P = P.tocsr()
        rng = np.random.default_rng(self.random_state)
        self.user_factors = rng.normal(0, 0.01, (P.shape[0], self.n_factors))
        self.item_factors = rng.normal(0, 0.01, (P.shape[1], self.n_factors))

        for _ in range(self.n_iterations):
            self.user_factors = self.solve(P, self.item_factors)
            self.item_factors = self.solve(P.T.tocsr(), self.user_factors)
        return self

    def solve(self, P, Y):
        YtY = Y.T @ Y + self.regularization * np.eye(self.n_factors)
        X = np.zeros((P.shape[0], self.n_factors))

        for row in range(P.shape[0]):
            start, end = P.indptr[row], P.indptr[row + 1]
            Yi = Y[P.indices[start:end]]
            targets = P.data[start:end]
            A = YtY + self.alpha * Yi.T @ Yi
            b = (1 + self.alpha) * Yi.T @ targets
            X[row] = np.linalg.solve(A, b)
        return X

    def score(self, users):
        return self.user_factors[users] @ self.item_factors.T
