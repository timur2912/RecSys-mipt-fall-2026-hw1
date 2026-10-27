import numpy as np
from scipy.sparse import csr_matrix

from models.als import ImplicitALS


class CustomTargetALS(ImplicitALS):

    def __init__(self, half_life=30.0, **als_params):
        super().__init__(**als_params)
        self.half_life = half_life

    def build_target(self, df, shape):
        positive = df[(df["rating"] > 3) & df["item_idx"].notna()]
        rating_signal = np.clip((positive["rating"] - 3) / 2, 0, 1)

        age = df["timestamp"].max() - positive["timestamp"]
        if hasattr(age, "dt"):
            age = age.dt.total_seconds() / 86400
        time_weight = np.exp(-np.log(2) * age / self.half_life)

        values = (rating_signal * time_weight).to_numpy(dtype=float)
        rows = positive["user_idx"].to_numpy(dtype=int)
        cols = positive["item_idx"].to_numpy(dtype=int)
        return csr_matrix((values, (rows, cols)), shape=shape)

    def fit(self, df, shape):
        return super().fit(self.build_target(df, shape))
