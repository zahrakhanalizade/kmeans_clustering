from typing import List, Tuple

import numpy as np

from utils import problem


@problem.tag("hw4-A")
def calculate_centers(
    data: np.ndarray, classifications: np.ndarray, num_centers: int
) -> np.ndarray:
    """
    Sub-routine of Lloyd's algorithm that calculates the centers given datapoints and their respective classifications/assignments.
    num_centers is additionally provided for speed-up purposes.

    Args:
        data (np.ndarray): Array of shape (n, d). Training data set.
        classifications (np.ndarray): Array of shape (n,) full of integers in range {0, 1, ...,  num_centers - 1}.
            Data point at index i is assigned to classifications[i].
        num_centers (int): Number of centers for reference.
            Might be usefull for pre-allocating numpy array (Faster that appending to list).

    Returns:
        np.ndarray: Array of shape (num_centers, d) containing new centers.

    Methods:
        This can be done either by looping across points or looping across centers.
        Looping across centers shows to be faster.
    """

    # looping across data points:
    # d = data.shape[1]
    # centers = np.zeros((num_centers, d))
    
    # count = np.zeros(num_centers)  # Number of data points assigned to each center
    # for i, datapoint in enumerate(data):
    #     center_idx = classifications[i]
    #     centers[center_idx] += datapoint
    #     count[center_idx] += 1

    # # Divide the summed values by the count to calculate the mean
    # centers /= count.reshape(-1, 1)
    
    # return centers


    # looping across centers:
    centers = np.zeros((num_centers, data.shape[1]))
    for k in range(num_centers):
        centers[k,:] = data[classifications == k,:].mean(axis=0)
    return centers
    


@problem.tag("hw4-A")
def cluster_data(data: np.ndarray, centers: np.ndarray) -> np.ndarray:
    """
    Sub-routine of Lloyd's algorithm that clusters datapoints to centers given datapoints and centers.

    Args:
        data (np.ndarray): Array of shape (n, d). Training data set.
        centers (np.ndarray): Array of shape (k, d). Each row is a center to which a datapoint can be clustered.

    Returns:
        np.ndarray: Array of integers of shape (n,), with each entry being in range {0, 1, 2, ..., k - 1}.
            Entry j at index i should mean that j^th center is the closest to data[i] datapoint.
    """
    distances = np.zeros((data.shape[0], centers.shape[0]))
    for idx, center in enumerate(centers):
        distances[:, idx] = np.sum((data - center) ** 2, axis=1)
    
    clusters = distances.argmin(axis=1)
    return clusters


# @problem.tag("hw4-A")
def calculate_error(data: np.ndarray, centers: np.ndarray) -> float:
    """Calculates error/objective function on a provided dataset, with trained centers.

    Args:
        data (np.ndarray): Array of shape (n, d). Dataset to evaluate centers on.
        centers (np.ndarray): Array of shape (k, d). Each row is a center to which a datapoint can be clustered.
            These should be trained on training dataset.

    Returns:
        float: Single value representing mean objective function of centers on a provided dataset.
    """
    distances = np.zeros((data.shape[0], centers.shape[0]))
    for idx, center in enumerate(centers):
        distances[:, idx] = np.sqrt(np.sum((data - center) ** 2, axis=1))
    return np.mean(np.min(distances, axis=1))


@problem.tag("hw4-A")
def lloyd_algorithm(
    data: np.ndarray, num_centers: int, epsilon: float = 10e-3
) -> Tuple[np.ndarray, List[float]]:
    """Main part of Lloyd's Algorithm.

    Args:
        data (np.ndarray): Array of shape (n, d). Training data set.
        num_centers (int): Number of centers to train/cluster around.
        epsilon (float, optional): Epsilon for stopping condition.
            Training should stop when max(abs(centers - previous_centers)) is smaller or equal to epsilon.
            Defaults to 10e-3.

    Returns:
        np.ndarray: Tuple of 2 numpy arrays:
            Element at index 0: Array of shape (num_centers, d) containing trained centers.
            Element at index 1: List of floats of length # of iterations
                containing errors at the end of each iteration of lloyd's algorithm.
                You should use the calculate_error() function that has been implemented for you.

    Note:
        - For initializing centers please use the first `num_centers` data points.
    """

    # initialize random centers:
    # min_values = np.min(data, axis=0)  # Minimum values along each dimension
    # max_values = np.max(data, axis=0)  # Maximum values along each dimension
    # centers = np.random.uniform(min_values, max_values, size=(num_centers, data.shape[1]))

    centers = data[0:num_centers,:]
    max_iter = 100
    errors = np.zeros((max_iter))

    for i in range(max_iter):
        classifications = cluster_data(data, centers)
        centers = calculate_centers(data, classifications, num_centers)
        error = calculate_error(data, centers)
        errors[i] = error
        if i == 0:
            continue
        if errors[i-1] - errors[i] < epsilon:
            break
    return centers, list(errors[0:(i+1)])

