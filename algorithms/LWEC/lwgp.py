import numpy as np

from sklearn.cluster import KMeans

from sklearn.metrics import pairwise_distances
import numpy as np
from scipy.cluster.hierarchy import linkage, fcluster
from sklearn.metrics import pairwise_distances
import numpy as np
from scipy.cluster.hierarchy import linkage, fcluster

from algorithms.LWEC.lwec import generate_base_clusterings, compute_eci


def locally_weighted_graph_partitioning(base_clusterings, n_clusters_list, n_objects, theta=0.4, k=4):
    # Compute uncertainty and ECI


    eci_matrix = compute_eci(base_clusterings, n_clusters_list, n_objects)
    lw_matrix = np.zeros((n_objects, n_objects))

    for t in range(len(base_clusterings)):
        labels = base_clusterings[t]
        for i in range(n_objects):
            for j in range(n_objects):
                if labels[i] == labels[j]:
                    lw_matrix[i, j] += eci_matrix[i, t] * eci_matrix[j, t]

    lw_matrix /= len(base_clusterings)

    dist = 1 - lw_matrix
    Z = linkage(pairwise_distances(dist), method='average')
    return fcluster(Z, k, criterion='maxclust') - 1



def do_lwgp(X, k):

    base_clusterings, n_clusters_list = generate_base_clusterings(X)
    labels = locally_weighted_graph_partitioning(base_clusterings, n_clusters_list, X.shape[0], 0.5, k)
    return labels