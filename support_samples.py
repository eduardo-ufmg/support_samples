import numpy as np
from sklearn.neighbors import NearestNeighbors, BallTree

def support_samples(X: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Return the subset of points from X (and their labels) that lie on at least one 
    Gabriel-graph edge connecting different labels (the support graph). 
    Uses a k-NN + range-search heuristic for efficiency.
    """
    X = np.asarray(X)
    y = np.asarray(y)
    # Ensure X is 2D
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    n_samples, dim = X.shape
    # Handle trivial cases
    if n_samples < 2 or np.unique(y).size <= 1:
        # No cross-class edges possible
        return np.empty((0, dim)), np.empty((0,), dtype=y.dtype)

    # Choose number of neighbors (heuristic)
    K = min(n_samples - 1, 30)
    # Build nearest-neighbor structure
    nbrs = NearestNeighbors(n_neighbors=K+1, algorithm='auto', metric='euclidean')
    nbrs.fit(X)
    distances, indices = nbrs.kneighbors(X, return_distance=True)
    
    # Build a BallTree for efficient radius queries (empty-ball checks)
    tree = BallTree(X)
    support_mask = np.zeros(n_samples, dtype=bool)
    
    # Check each point's neighbors for valid support edges
    for i in range(n_samples):
        xi = X[i]
        yi = y[i]
        # Skip the first neighbor (itself) by starting at 1
        for k in range(1, distances.shape[1]):
            j = indices[i, k]
            # Avoid duplicate (i,j) vs (j,i) by requiring j>i
            if j <= i:
                continue
            if y[j] == yi:
                continue  # same label, not a support edge
            # Compute midpoint and radius of the candidate edge
            center = (xi + X[j]) * 0.5
            radius = distances[i, k] * 0.5
            # Query points within radius of center
            neighbors_idx = tree.query_radius(center.reshape(1, -1), r=radius)[0]
            # If no other point (besides i,j) is inside the ball, accept the edge
            # (BallTree is inclusive of the boundary, so i and j themselves appear if within radius)
            valid = True
            for m in neighbors_idx:
                if m != i and m != j:
                    valid = False
                    break
            if valid:
                support_mask[i] = True
                support_mask[j] = True

    # Gather support samples
    support_indices = np.nonzero(support_mask)[0]
    X_support = X[support_indices]
    y_support = y[support_indices]
    return X_support, y_support
