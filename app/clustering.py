
import numpy as np
from sklearn.cluster import (
    KMeans, DBSCAN, AgglomerativeClustering,
    SpectralClustering, MeanShift
)
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score

def get_default_clusters(dataset_name):
    if dataset_name in ["Blobs", "Iris"]:
        return 3
    elif dataset_name in ["Moons", "Circles"]:
        return 2
    elif dataset_name in ["MNIST"]:
        return 10
    return 3

def perform_clustering(X, algorithm, n_clusters):
    if algorithm == "KMeans":
        model = KMeans(n_clusters=n_clusters, random_state=42)
        labels = model.fit_predict(X)
        centers = model.cluster_centers_
    elif algorithm == "DBSCAN":
        model = DBSCAN(eps=0.3, min_samples=5)
        labels = model.fit_predict(X)
        centers = None
    elif algorithm == "AgglomerativeClustering":
        model = AgglomerativeClustering(n_clusters=n_clusters)
        labels = model.fit_predict(X)
        centers = None
    elif algorithm == "GaussianMixture":
        model = GaussianMixture(n_components=n_clusters, random_state=42)
        model.fit(X)
        labels = model.predict(X)
        centers = model.means_
    elif algorithm == "SpectralClustering":
        model = SpectralClustering(n_clusters=n_clusters, random_state=42, assign_labels='discretize')
        labels = model.fit_predict(X)
        centers = None
    elif algorithm == "MeanShift":
        model = MeanShift()
        labels = model.fit_predict(X)
        centers = model.cluster_centers_
    else:
        labels = np.zeros(X.shape[0])
        centers = None

    return labels, centers

def compute_silhouette(X, labels):
    unique_labels = set(labels)
    if len(unique_labels - {-1}) > 1:
        return silhouette_score(X, labels)
    return None
