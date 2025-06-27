from itertools import combinations
from warnings import warn

import numpy as np
from scipy.spatial import Delaunay, cKDTree  # type: ignore
from sklearn.cluster import KMeans
from sklearn.neighbors import NearestNeighbors


def support_samples(
    X: np.ndarray, y: np.ndarray, method: str = "hnbf"
) -> tuple[np.ndarray, np.ndarray]:
    """
    Select a compact set of boundary (support) samples from labeled data.

    Parameters:
    X : np.ndarray, shape (n_samples, n_features)
        Feature array.
    y : np.ndarray, shape (n_samples,)
        Class labels for each row in X.
    method : str, optional ['hnbf', 'margin_clustering', 'gabriel_graph']
        Method to use for selecting support samples. Options are 'hnbf' for
        Heterogeneous-Neighborhood Boundary Filtering, 'margin_clustering'
        for Margin Clustering method, and 'gabriel_graph' for Gabriel Graph method.
        Default is 'hnbf'.

    Returns:
    X_support : np.ndarray
        Selected support points (subset of X).
    y_support : np.ndarray
        Corresponding labels of X_support.
    """
    if method == "hnbf":
        return hnbf(X, y)
    elif method == "margin_clustering":
        return margin_clustering(X, y)
    elif method == "gabriel_graph":
        return gabriel_graph(X, y)
    else:
        raise ValueError(f"Unknown method: {method}")


def hnbf(X: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
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

    Parameters:
    X : np.ndarray
        A 2D numpy array of shape (n_samples, n_features) containing the
        feature data.
    y : np.ndarray
        A 1D numpy array of shape (n_samples,) containing the integer class
        labels for each sample.

    Returns:
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

    if k == 0:  # Handle case with only one sample
        return np.empty((0, X.shape[1])), np.empty((0,))

    # Use BallTree for efficient search in moderate-to-high dimensions.
    # Query for k+1 neighbors to account for the point itself being returned.
    nn_model = NearestNeighbors(n_neighbors=k + 1, algorithm="ball_tree", n_jobs=-1)
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
    is_heterogeneous = np.any(neighbor_labels != point_labels, axis=1)

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
    num_different_neighbors = np.sum(
        candidate_neighbor_labels != candidate_point_labels, axis=1
    )

    # Calculate the Heterogeneity Score (H) for all candidates.
    heterogeneity_score = num_different_neighbors / k

    # Apply the non-arbitrary threshold to filter out points in overlap regions.
    # We keep points with 0 < H < 0.5.
    # The condition `num_different_neighbors > 0` is already guaranteed by Stage 1.
    final_selection_mask = heterogeneity_score < 0.5

    final_indices = candidate_indices[0][final_selection_mask]

    # Return the subset of data and labels corresponding to the final indices.
    return X[final_indices], y[final_indices]


def margin_clustering(X: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Select a compact set of boundary (support) samples from labeled data.

    Parameters:
    X : np.ndarray, shape (n_samples, n_features)
        Feature array.
    y : np.ndarray, shape (n_samples,)
        Class labels for each row in X.

    Returns:
    X_support : np.ndarray
        Selected support points (subset of X).
    y_support : np.ndarray
        Corresponding labels of X_support.
    """
    X = np.asarray(X)
    y = np.asarray(y)
    unique_classes = np.unique(y)
    support_idx = set()

    for cls in unique_classes:
        idx_cls = np.where(y == cls)[0]
        X_cls = X[idx_cls]
        n_cls = len(idx_cls)
        if n_cls == 0:
            continue

        # Distance to nearest same-class neighbor:
        if n_cls > 1:
            nbrs_same = NearestNeighbors(n_neighbors=2, algorithm="auto").fit(X_cls)
            dist_same, _ = nbrs_same.kneighbors(X_cls)
            d_same = dist_same[:, 1]  # first neighbor is self (distance 0)
        else:
            d_same = np.array([0.0])  # only one point in class

        # Distance to nearest different-class neighbor:
        other_idx = np.where(y != cls)[0]
        if len(other_idx) > 0:
            nbrs_diff = NearestNeighbors(n_neighbors=1, algorithm="auto").fit(
                X[other_idx]
            )
            dist_diff, ind_diff = nbrs_diff.kneighbors(X_cls)
            d_diff = dist_diff[:, 0]
            neighbor_idx = other_idx[ind_diff[:, 0]]
        else:
            # If this is the only class, skip
            continue

        # Compute margin and select positive margins
        margin = d_diff - d_same
        pos_mask = margin > 0
        if np.sum(pos_mask) == 0:
            continue

        pos_margin = margin[pos_mask]
        global_idx = idx_cls[pos_mask]
        neighbors = neighbor_idx[pos_mask]

        # Cluster the positive margins into two groups
        # (Cluster 0 will be the smaller-margin "boundary" group)
        if len(pos_margin) > 1:
            km = KMeans(n_clusters=2, random_state=0).fit(pos_margin.reshape(-1, 1))
            labels = km.labels_
            centers = km.cluster_centers_.flatten()
            boundary_label = np.argmin(centers)
            chosen = labels == boundary_label
        else:
            chosen = np.ones(len(pos_margin), dtype=bool)

        # Add chosen points and their opposite-class neighbors
        chosen_idx = global_idx[chosen]
        chosen_neighbors = neighbors[chosen]
        for i_pt, j_pt in zip(chosen_idx, chosen_neighbors):
            support_idx.add(int(i_pt))
            if j_pt != -1:
                support_idx.add(int(j_pt))

    # Return the selected support points
    support_idx = np.array(sorted(support_idx), dtype=int)
    X_support = X[support_idx]
    y_support = y[support_idx]
    return X_support, y_support


def gabriel_graph_bruteforce(
    X: np.ndarray, y: np.ndarray, tree: cKDTree
) -> tuple[np.ndarray, np.ndarray]:
    """
    A helper function to compute the inter-class Gabriel Graph vertices via a
    brute-force O(n^2 * log n) method. It checks every pair of points.

    Parameters:
        X (np.ndarray): The matrix of points.
        y (np.ndarray): The array of labels.
        tree (cKDTree): A pre-computed KD-Tree on X for efficient queries.

    Returns:
        The filtered points and their labels.
    """
    n_points, n_dims = X.shape
    final_vertex_indices = set()

    # Iterate over all unique pairs of points
    for i, j in combinations(range(n_points), 2):
        # Optimization: Check the cheaper inter-class condition first
        if y[i] != y[j]:
            # Verify the Gabriel property for the edge (i, j)
            p_i, p_j = X[i], X[j]
            midpoint = (p_i + p_j) / 2.0
            radius = np.linalg.norm(p_i - midpoint)

            # Query for points within the diametral sphere
            points_in_sphere = tree.query_ball_point(midpoint, radius + 1e-9)

            # If no point other than i or j is in the sphere, it's a Gabriel edge
            is_gabriel = True
            for k in points_in_sphere:
                if k != i and k != j:
                    is_gabriel = False
                    break

            if is_gabriel:
                final_vertex_indices.add(i)
                final_vertex_indices.add(j)

    if not final_vertex_indices:
        return np.empty((0, n_dims), dtype=X.dtype), np.empty(0, dtype=y.dtype)

    indices = np.fromiter(
        final_vertex_indices, dtype=int, count=len(final_vertex_indices)
    )
    indices.sort()

    return X[indices], y[indices]


def gabriel_graph(X: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Computes the Gabriel Graph for a set of points, then extracts vertices
    that belong to edges connecting points of different classes.

    This implementation prioritizes performance and memory efficiency. It uses
    scipy.spatial.Delaunay to find candidate edges and scipy.spatial.cKDTree
    for efficient geometric queries to filter for Gabriel edges.

    Parameters:
        X (np.ndarray): A matrix of shape (n_points, n_dims) where each
                        row is a point and each column is a coordinate.
        y (np.ndarray): An array of shape (n_points,) containing the class
                        label for each point.

    Returns:
        tuple[np.ndarray, np.ndarray]: A tuple containing two arrays:
            - The first is a subset of X containing only the points that are
              vertices of at least one inter-class Gabriel edge.
            - The second is the corresponding subset of y.
    """
    n_points, n_dims = X.shape
    if n_points < 2:
        return np.empty((0, n_dims), dtype=X.dtype), np.empty(0, dtype=y.dtype)

    # For performance, Delaunay triangulation is used to generate a small
    # superset of the Gabriel graph edges. This is much faster than checking
    # all O(n^2) pairs, especially in low dimensions.
    # The 'QJ' option tells qhull to jitter the input, which can help
    # prevent precision issues with co-spherical or co-planar points.
    try:
        delaunay = Delaunay(X, qhull_options="QJ")
    except Exception as e:
        # Delaunay can fail in high dimensions or with degenerate data.
        # Given the focus on performance, we do not include a slower fallback
        # and instead notify the user of the failure.
        warn(
            f"Delaunay triangulation failed: {e}. "
            "This can happen in high dimensions or with degenerate input data. "
            "Falling back to a slower O(n^2 * log n) brute-force method.",
            UserWarning,
        )
        tree = cKDTree(X)
        return gabriel_graph_bruteforce(X, y, tree)

    # Extract unique edges from the Delaunay simplices.
    # Using a set of sorted tuples ensures each edge is represented once.
    candidate_edges = set()
    for simplex in delaunay.simplices:
        for i, j in combinations(simplex, 2):
            candidate_edges.add(tuple(sorted((i, j))))

    # A KD-Tree provides fast spatial searches, which we'll use to efficiently
    # check the Gabriel graph condition for each candidate edge.
    tree = cKDTree(X)

    # This set will store the indices of vertices that we need to keep.
    final_vertex_indices = set()

    # Process each candidate edge.
    for i, j in candidate_edges:
        # Optimization: Check for inter-class connection first, as it's a cheap operation.
        if y[i] != y[j]:
            # The edge connects different classes. Now, verify the Gabriel property:
            # The closed sphere with diameter (X[i], X[j]) must contain no other point from X.
            p_i, p_j = X[i], X[j]
            midpoint = (p_i + p_j) / 2.0
            radius = np.linalg.norm(p_i - midpoint)

            # Query for any points within the diametral sphere. The query_ball_point
            # method checks for distances <= radius.
            points_in_sphere = tree.query_ball_point(
                midpoint, radius + 1e-9
            )  # Add epsilon for float safety

            # Check if any point other than i or j is in the sphere.
            is_gabriel = True
            for k in points_in_sphere:
                if k != i and k != j:
                    is_gabriel = False
                    break

            if is_gabriel:
                # This edge is an inter-class Gabriel edge. Add its vertices to our final set.
                final_vertex_indices.add(i)
                final_vertex_indices.add(j)

    if not final_vertex_indices:
        return np.empty((0, n_dims), dtype=X.dtype), np.empty(0, dtype=y.dtype)

    # Convert the set of indices to a sorted numpy array for slicing.
    # np.fromiter is generally faster than list conversion for this task.
    indices = np.fromiter(
        final_vertex_indices, dtype=int, count=len(final_vertex_indices)
    )
    indices.sort()  # Sorting provides deterministic output.

    return X[indices], y[indices]
