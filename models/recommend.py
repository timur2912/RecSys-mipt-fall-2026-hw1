import numpy as np


class RecommenderMixin:
    def recommend(self, user_idx, seen_matrix, n=10):
        return self.recommend_batch([user_idx], seen_matrix, n)[0]

    def recommend_batch(self, user_indices, seen_matrix, n=10, user_batch_size=128):
        result = []
        for start in range(0, len(user_indices), user_batch_size):
            users = np.asarray(user_indices[start : start + user_batch_size])
            scores = self.score(users)
            scores[seen_matrix[users].toarray() > 0] = -np.inf
            result.append(np.argsort(-scores, axis=1, kind="stable")[:, :n])
        return np.vstack(result)
