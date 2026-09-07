import numpy as np
from itertools import combinations

class CombinatorialPurgedCV:
    def __init__(self, n_groups: int = 6, n_test_groups: int = 2, embargo_pct: float = 0.01):
        self.n_groups = n_groups
        self.n_test_groups = n_test_groups
        self.embargo_pct = embargo_pct

    def split(self, n_samples: int, label_end_idx: np.ndarray = None):
        group_bounds = np.linspace(0, n_samples, self.n_groups + 1).astype(int)
        groups = [np.arange(group_bounds[i], group_bounds[i + 1]) for i in range(self.n_groups)]
        embargo = int(n_samples * self.embargo_pct)

        for test_group_ids in combinations(range(self.n_groups), self.n_test_groups):
            test_idx = np.concatenate([groups[g] for g in test_group_ids])
            test_start, test_end = int(test_idx.min()), int(test_idx.max())

            train_idx = np.concatenate([groups[g] for g in range(self.n_groups) if g not in test_group_ids])

            if label_end_idx is not None:
                ends = np.nan_to_num(label_end_idx[train_idx], nan=train_idx)
                overlap_mask = (ends >= test_start) & (train_idx <= test_end)
                train_idx = train_idx[~overlap_mask]

            embargo_mask = (train_idx > test_end) & (train_idx <= test_end + embargo)
            train_idx = train_idx[~embargo_mask]

            yield train_idx, test_idx, test_group_ids
