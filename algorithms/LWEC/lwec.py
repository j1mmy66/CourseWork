
import numpy as np

from sklearn.cluster import KMeans

from sklearn.metrics import pairwise_distances
import numpy as np
from scipy.cluster.hierarchy import linkage, fcluster
from sklearn.metrics import pairwise_distances
import numpy as np
from scipy.cluster.hierarchy import linkage, fcluster





def generate_base_clusterings(X, n_runs=5):
    base_clusterings = []
    n_clusters_list = []
    for _ in range(n_runs):
        k = np.random.randint(2, 10)
        labels = KMeans(n_clusters=k, random_state=np.random.randint(10000)).fit_predict(X)
        base_clusterings.append(labels)
        n_clusters_list.append(k)
    return base_clusterings, n_clusters_list

def entropy(labels):
    _, counts = np.unique(labels, return_counts=True)
    probs = counts / counts.sum()
    return -np.sum(probs * np.log2(probs))

def cluster_uncertainty(cluster_labels):
    return entropy(cluster_labels)

def compute_eci(base_clusterings, n_clusters_list, n_objects):
    n_partitions = len(base_clusterings)
    eci = np.zeros((n_objects, n_partitions))
    for i, labels in enumerate(base_clusterings):
        for j in range(n_objects):
            cluster_label = labels[j]
            mask = labels == cluster_label
            eci[j, i] = 1 - cluster_uncertainty(labels[mask]) / np.log2(n_clusters_list[i])
    return eci

