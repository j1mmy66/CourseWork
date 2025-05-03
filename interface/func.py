from clustering.clustering_sklearn import get_default_clusters, perform_clustering, compute_silhouette
from clustering.metrics import compute_davies_bouldin, compute_calinski_harabasz, compute_adjusted_rand, compute_nmi, \
    compute_hcv, compute_silhouette
from data.datasets import load_blobs, load_circles, load_mnist_from_db, load_moons
from clustering.plot_utils import save_cluster_plot
from generator.generator import generate_synthetic_data

datasets_funcs = {
    "Blobs": load_blobs,
    "Moons": load_moons,
    "Circles": load_circles,
    "MNIST": load_mnist_from_db
}


def apply_clustering_or_generate(
        source: str,
        dataset_name: str,
        N: int, V: int, K_star: int, n_min: int, alpha: float,
        algorithm: str
):
    if source == "Сгенерировать данные":
        X, y_true = generate_synthetic_data(N, V, K_star, n_min, alpha)
    else:
        X, y_true = datasets_funcs[dataset_name]()
    if source == "Сгенерировать данные":
        n_clusters = K_star
    else:
        n_clusters = get_default_clusters(dataset_name)
    labels, centers = perform_clustering(X, algorithm, n_clusters)
    # Внутренние метрики (без y_true)
    silhouette = compute_silhouette(X, labels)
    davies_bouldin = compute_davies_bouldin(X, labels)
    calinski_harabasz = compute_calinski_harabasz(X, labels)

    # Внешние метрики (с y_true)
    adjusted_rand = compute_adjusted_rand(y_true, labels)
    nmi = compute_nmi(y_true, labels)
    homogeneity, completeness, v_measure = compute_hcv(y_true, labels)

    return save_cluster_plot(X, labels, algorithm, dataset_name, centers, "cluster_plot.png")

