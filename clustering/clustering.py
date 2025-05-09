
import numpy as np
from sklearn.cluster import (
    KMeans, DBSCAN, AgglomerativeClustering,
    SpectralClustering, MeanShift
)
from sklearn.mixture import GaussianMixture

from algorithms.ACMK.acmk import acmk_clustering
from algorithms.LWEC.lwea import do_lwea
from algorithms.LWEC.lwgp import do_lwgp
from algorithms.USPEC.usenc import usenc

from algorithms.USPEC.uspec import uspec


def get_default_clusters(dataset_name):
    if dataset_name in ["Blobs", "Iris"]:
        return 3
    elif dataset_name in ["Moons", "Circles"]:
        return 2
    elif dataset_name in ["Digits"]:
        return 10
    return 3

def perform_clustering(X, algorithm, n_clusters):
    if algorithm == "USENC":
        labels = usenc(X, n_clusters)

    elif algorithm == "USPEC":
        labels = uspec(X, n_clusters)

    elif algorithm == "KMeans":
        model = KMeans(n_clusters=n_clusters, random_state=42)
        labels = model.fit_predict(X)

    elif algorithm == "DBSCAN":
        model = DBSCAN(eps=0.3, min_samples=5)
        labels = model.fit_predict(X)

    elif algorithm == "AgglomerativeClustering":
        model = AgglomerativeClustering(n_clusters=n_clusters)
        labels = model.fit_predict(X)

    elif algorithm == "GaussianMixture":
        model = GaussianMixture(n_components=n_clusters, random_state=42)
        model.fit(X)
        labels = model.predict(X)
        centers = model.means_
    elif algorithm == "SpectralClustering":
        model = SpectralClustering(n_clusters=n_clusters, random_state=42, assign_labels='discretize')
        labels = model.fit_predict(X)

    elif algorithm == "MeanShift":
        model = MeanShift()
        labels = model.fit_predict(X)

    elif algorithm == "LWEA":
        labels = do_lwea(X, n_clusters)

    elif algorithm == "LWGP":
        labels = do_lwgp(X, n_clusters)
    elif algorithm == "ACMK":
        labels, _ = acmk_clustering(X, n_clusters)
    else:
        labels = np.zeros(X.shape[0])


    return labels


