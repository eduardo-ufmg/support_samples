import numpy as np
from sklearn.neighbors import BallTree

def support_samples(X: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Condensed Nearest-Neighbor prototype selection (non-parametric).

    Parameters:
        X : np.ndarray, shape (n_samples, n_features)
            Feature matrix.
        y : np.ndarray, shape (n_samples,)
            Integer or string labels.

    Returns:
        X_proto : np.ndarray, shape (n_prototypes, n_features)
        y_proto : np.ndarray, shape (n_prototypes,)
            The subset of X and y that forms the condensed prototype set.
    """
    n_samples = X.shape[0]
    # 1) Initialize with one example per class:
    #    pick the first occurrence of each label
    classes, first_idx = np.unique(y, return_index=True)
    S_idx = list(first_idx)
    selected = np.zeros(n_samples, dtype=bool)
    selected[S_idx] = True

    # 2) Iteratively add *all* points mis‑classified by current prototypes
    while True:
        # Build BallTree on current prototypes
        tree = BallTree(X[S_idx])
        # Indices of points not yet in S
        unselected_idx = np.nonzero(~selected)[0]
        if unselected_idx.size == 0:
            break

        # Query their nearest prototype
        dist, nn = tree.query(X[unselected_idx], k=1)
        nn_global = np.array(S_idx, dtype=int)[nn.ravel()]

        # Find those whose nearest has a different label
        mask_mis = y[nn_global] != y[unselected_idx]
        if not np.any(mask_mis):
            break

        # Add all mis‑classified at once
        new_idx = unselected_idx[mask_mis]
        S_idx.extend(new_idx.tolist())
        selected[new_idx] = True

    # 3) Return prototypes
    S_idx = np.array(S_idx, dtype=int)
    return X[S_idx], y[S_idx]
