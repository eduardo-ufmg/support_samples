import numpy as np
from sklearn.neighbors import NearestNeighbors

def support_samples(X: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Identifies a compact subset of samples that lie on the boundaries between
    class distributions using the Heterogeneous-Neighborhood Boundary Filtering
    (HNBF) algorithm.

    This non-parametric method is designed to be efficient and scalable for large
    datasets with moderate to high dimensionality. It operates in two stages:
    1.  Condensing: It first identifies a candidate set of all non-interior
        points by checking for local neighborhood heterogeneity. A point is
        considered a candidate if its neighborhood contains points from more
        than one class.
    2.  Editing: It then filters this candidate set by computing a
        Heterogeneity Score for each point. Points with high scores (>= 0.5),
        indicative of being in ambiguous or overlapping regions, are removed.

    The number of neighbors, k, is determined programmatically from the data
    as the nearest odd integer to the square root of the number of samples (N).
    The neighbor search is performed using a BallTree for optimal performance
    in high-dimensional spaces.

    Parameters
    ----------
    X : np.ndarray
        A 2D numpy array of shape (n_samples, n_features) containing the
        feature data.
    y : np.ndarray
        A 1D numpy array of shape (n_samples,) containing the integer class
        labels for each sample.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        A tuple containing two numpy arrays:
        - The first array is the subset of X corresponding to the selected
          support samples.
        - The second array is the subset of y corresponding to the selected
          support samples.
    """
    n_samples = X.shape[0]
    if n_samples == 0:
        return np.empty((0, X.shape[1])), np.empty((0,))

    # Data-driven calculation of k.
    k = int(np.sqrt(n_samples))
    if k % 2 == 0:
        k += 1
    
    # Ensure k is at least 1 and not larger than the number of samples
    k = max(1, min(k, n_samples - 1))  # Ensure k is valid

    if k == 0: # Handle case with only one sample
        return np.empty((0, X.shape[1])), np.empty((0,))
    
    # Use BallTree for efficient search in moderate-to-high dimensions.
    # Query for k+1 neighbors to account for the point itself being returned.
    nn_model = NearestNeighbors(n_neighbors=k + 1, algorithm='ball_tree', n_jobs=-1)
    nn_model.fit(X)
    _, indices = nn_model.kneighbors(X)

    # Exclude the first neighbor (the point itself) from the analysis.
    neighbor_indices = indices[:, 1:]

    # Get the labels of all neighbors for all points in a single vectorized operation.
    neighbor_labels = y[neighbor_indices]
    
    # Get the labels of the points themselves, reshaped for broadcasting.
    point_labels = y.reshape(-1, 1)

    # --- Stage 1: Condensing via Homogeneity Check ---
    # A point is heterogeneous if at least one neighbor has a different label.
    # This is a vectorized check across all points.
    is_heterogeneous = np.any(neighbor_labels!= point_labels, axis=1)
    
    # The candidate set contains indices of all heterogeneous points.
    candidate_indices = np.where(is_heterogeneous)

    if candidate_indices[0].size == 0:
        return np.empty((0, X.shape[1])), np.empty((0,))

    # --- Stage 2: Editing via Heterogeneity Thresholding ---
    # Focus only on the candidate points for the next stage.
    candidate_neighbor_labels = neighbor_labels[candidate_indices]
    candidate_point_labels = point_labels[candidate_indices]

    # Calculate the number of neighbors with a different class for each candidate.
    # This is also fully vectorized.
    num_different_neighbors = np.sum(candidate_neighbor_labels!= candidate_point_labels, axis=1)
    
    # Calculate the Heterogeneity Score (H) for all candidates.
    heterogeneity_score = num_different_neighbors / k

    # Apply the non-arbitrary threshold to filter out points in overlap regions.
    # We keep points with 0 < H < 0.5.
    # The condition `num_different_neighbors > 0` is already guaranteed by Stage 1.
    final_selection_mask = heterogeneity_score < 0.5
    
    final_indices = candidate_indices[0][final_selection_mask]

    # Return the subset of data and labels corresponding to the final indices.
    return X[final_indices], y[final_indices]