import numpy as np

def pdist2_fast(X, Y, metric='sqeuclidean'):
    """
    Calculates the distance between sets of vectors.
    Parameters:
        X (ndarray): Matrix of shape (m, d), where m is the number of vectors.
        Y (ndarray): Matrix of shape (n, d), where n is the number of vectors.
        metric (str): Distance metric to use. Options are:
                      'sqeuclidean', 'euclidean', 'L1', 'cosine', 'emd', 'chisq'.

    Returns:
        D (ndarray): Distance matrix of shape (m, n).
    """
    if metric in [0, 'sqeuclidean']:
        D = dist_euc_sq(X, Y)
    elif metric == 'euclidean':
        D = np.sqrt(dist_euc_sq(X, Y))
    elif metric == 'L1':
        D = dist_l1(X, Y)
    elif metric == 'cosine':
        D = dist_cosine(X, Y)
    elif metric == 'emd':
        D = dist_emd(X, Y)
    elif metric == 'chisq':
        D = dist_chisq(X, Y)
    else:
        raise ValueError(f"Unknown metric: {metric}")
    return np.maximum(0, D)

def dist_l1(X, Y):
    """
    L1 (Manhattan) distance.
    """
    m, n = X.shape[0], Y.shape[0]
    D = np.zeros((m, n))
    for i in range(n):
        yi = np.tile(Y[i, :], (m, 1))
        D[:, i] = np.sum(np.abs(X - yi), axis=1)
    return D

def dist_cosine(X, Y):
    """
    Cosine distance.
    """
    X_norm = X / np.sqrt(np.sum(X**2, axis=1, keepdims=True))
    Y_norm = Y / np.sqrt(np.sum(Y**2, axis=1, keepdims=True))
    return 1 - np.dot(X_norm, Y_norm.T)

def dist_emd(X, Y):
    """
    Earth Mover's Distance (EMD).
    """
    Xcdf = np.cumsum(X, axis=1)
    Ycdf = np.cumsum(Y, axis=1)
    m, n = X.shape[0], Y.shape[0]
    D = np.zeros((m, n))
    for i in range(n):
        ycdf = np.tile(Ycdf[i, :], (m, 1))
        D[:, i] = np.sum(np.abs(Xcdf - ycdf), axis=1)
    return D

def dist_chisq(X, Y):
    """
    Chi-squared distance.
    """
    m, n = X.shape[0], Y.shape[0]
    D = np.zeros((m, n))
    for i in range(n):
        yi = np.tile(Y[i, :], (m, 1))
        s = yi + X
        d = yi - X
        D[:, i] = np.sum(d**2 / (s + np.finfo(float).eps), axis=1)
    return D / 2

def dist_euc_sq(X, Y):
    """
    Squared Euclidean distance.
    """
    XX = np.sum(X**2, axis=1, keepdims=True)
    YY = np.sum(Y**2, axis=1, keepdims=True).T
    return np.abs(XX + YY - 2 * np.dot(X, Y.T))
