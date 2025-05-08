
import numpy as np

from sklearn.cluster import KMeans





def locally_weighted_graph_partitioning(base_clusterings, n_clusters_list, n_objects, theta=0.4, k=4):
    from sklearn.metrics import pairwise_distances
    import numpy as np
    from scipy.cluster.hierarchy import linkage, fcluster

    # Compute uncertainty and ECI
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

def locally_weighted_evidence_accumulation(base_clusterings, n_clusters_list, n_objects, theta=0.4, k=4):
    from sklearn.metrics import pairwise_distances
    import numpy as np
    from scipy.cluster.hierarchy import linkage, fcluster

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

def generate_base_clusterings(X, n_runs=5):
    base_clusterings = []
    n_clusters_list = []
    for _ in range(n_runs):
        k = np.random.randint(2, 10)
        labels = KMeans(n_clusters=k, random_state=np.random.randint(10000)).fit_predict(X)
        base_clusterings.append(labels)
        n_clusters_list.append(k)
    return base_clusterings, n_clusters_list

def do_lwec(X, k_g):

    base_clusterings, n_clusters_list = generate_base_clusterings(X)
    labels = locally_weighted_graph_partitioning(base_clusterings, n_clusters_list, X.shape[0], theta=0.4, k=k_g)
    return labels

