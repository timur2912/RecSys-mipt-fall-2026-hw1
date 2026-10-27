import numpy as np
import torch
from torch import nn

from models.recommend import RecommenderMixin


class FactorizationMachine(nn.Module):
    def __init__(self, n_features, n_factors):
        super().__init__()
        self.bias = nn.Parameter(torch.zeros(1))
        self.w = nn.Embedding(n_features, 1)
        self.v = nn.Embedding(n_features, n_factors)
        nn.init.zeros_(self.w.weight)
        nn.init.normal_(self.v.weight, std=0.01)

    def forward(self, x):
        v = self.v(x)
        pairs = 0.5 * (v.sum(1) ** 2 - (v**2).sum(1)).sum(1)
        return self.bias + self.w(x).sum((1, 2)) + pairs


class FMRecommender(RecommenderMixin):
    def __init__(self, n_factors=16, learning_rate=0.01, weight_decay=1e-5, n_epochs=5,
                 batch_size=4096, negative_ratio=1, random_state=42, device="cpu"):
        self.n_factors = n_factors
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.negative_ratio = negative_ratio
        self.random_state = random_state
        self.device = device

    def features(self, users, items):
        x = np.column_stack([users, items, self.item_features[items]]) + self.offsets
        return torch.tensor(x, dtype=torch.long, device=self.device)

    def sample_negatives(self, users, seen, rng):
        items = rng.integers(self.n_items, size=len(users))
        bad = seen[users, items].A1 > 0
        while bad.any():
            items[bad] = rng.integers(self.n_items, size=bad.sum())
            bad = seen[users, items].A1 > 0
        return items

    def fit(self, P, seen, item_features):
        seen = seen.tocsr()
        self.n_items = P.shape[1]
        self.item_features = np.asarray(item_features)
        sizes = [P.shape[0], P.shape[1], *(self.item_features.max(axis=0) + 1)]
        self.offsets = np.cumsum([0, *sizes[:-1]])

        torch.manual_seed(self.random_state)
        rng = np.random.default_rng(self.random_state)
        self.model = FactorizationMachine(sum(sizes), self.n_factors).to(self.device)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate,
                                     weight_decay=self.weight_decay)
        loss_fn = nn.BCEWithLogitsLoss()
        pos_users, pos_items = P.nonzero()
        self.loss_history = []

        for _ in range(self.n_epochs):
            order = rng.permutation(len(pos_users))
            losses = []
            for start in range(0, len(order), self.batch_size):
                batch = order[start : start + self.batch_size]
                neg_users = np.repeat(pos_users[batch], self.negative_ratio)
                neg_items = self.sample_negatives(neg_users, seen, rng)

                x = self.features(np.r_[pos_users[batch], neg_users],
                                  np.r_[pos_items[batch], neg_items])
                y = torch.tensor(np.r_[np.ones(len(batch)), np.zeros(len(neg_users))],
                                 dtype=torch.float32, device=self.device)

                loss = loss_fn(self.model(x), y)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                losses.append(loss.item())
            self.loss_history.append(float(np.mean(losses)))
        return self

    def score(self, users):
        users = np.asarray(users)
        x = self.features(np.repeat(users, self.n_items), np.tile(np.arange(self.n_items), len(users)))
        with torch.no_grad():
            return self.model(x).reshape(len(users), -1).cpu().numpy()
