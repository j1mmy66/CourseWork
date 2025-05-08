from sklearn.metrics import pairwise_distances
import numpy as np
from scipy.cluster.hierarchy import linkage, fcluster

from algorithms.LWEC.lwec import compute_eci, generate_base_clusterings


def locally_weighted_evidence_accumulation(base_clusterings, n_clusters_list, n_objects, theta=0.4, k=4):




    eci_matrix = compute_eci(base_clusterings, n_clusters_list, n_objects)
    lwca_matrix = np.zeros((n_objects, n_objects))

    for t in range(len(base_clusterings)):
        labels = base_clusterings[t]
        for i in range(n_objects):
            for j in range(n_objects):
                if labels[i] == labels[j]:
                    lwca_matrix[i, j] += 1 + theta * (eci_matrix[i, t] + eci_matrix[j, t])

    lwca_matrix /= len(base_clusterings)
    dist = 1 - lwca_matrix
    Z = linkage(pairwise_distances(dist), method='average')
    return fcluster(Z, k, criterion='maxclust') - 1






def do_lwea(X, k):
    base_clusterings, n_clusters_list = generate_base_clusterings(X)

    labels = locally_weighted_evidence_accumulation(base_clusterings, n_clusters_list, X.shape[0], 0.4, k)

    return labels