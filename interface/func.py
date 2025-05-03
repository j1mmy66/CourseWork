import numpy as np

from clustering.clustering_sklearn import get_default_clusters, perform_clustering
from clustering.metrics import compute_davies_bouldin, compute_calinski_harabasz, compute_adjusted_rand, compute_nmi, \
    compute_hcv, compute_silhouette
from data.datasets import load_blobs, load_circles,  load_moons, load_digit
from clustering.plot_utils import save_cluster_plot
from data.db import insert_history_data
from generator.generator import generate_synthetic_data
import pandas as pd

datasets_funcs = {
    "Blobs": load_blobs,
    "Moons": load_moons,
    "Circles": load_circles,
    "Digits": load_digit
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

    metrics = [
        ["Silhouette", silhouette if silhouette is not None else np.nan],
        ["Davies–Bouldin", davies_bouldin if davies_bouldin is not None else np.nan],
        ["Calinski–Harabasz", calinski_harabasz if calinski_harabasz is not None else np.nan],
        ["Adjusted Rand", adjusted_rand],
        ["NMI", nmi],
        ["Homogeneity", homogeneity],
        ["Completeness", completeness],
        ["V‑Measure", v_measure]
    ]
    metrics_df = pd.DataFrame(metrics, columns=["Метрика", "Значение"])
    metrics_df["Значение"] = metrics_df["Значение"].apply(
        lambda x: "—" if x is None or (isinstance(x, float) and np.isnan(x))
        else f"{x:.3f}"
    )

    plot_path = save_cluster_plot(
        X, labels, algorithm, dataset_name, centers, "cluster_plot.png"
    )

    metrics_dict = metrics_df.set_index("Метрика")["Значение"].to_dict()

    # --- Вставляем в БД именно эти строки ---
    insert_history_data(
        dataset_name,
        algorithm,
        metrics_dict["Silhouette"],
        metrics_dict["Davies–Bouldin"],
        metrics_dict["Calinski–Harabasz"],
        metrics_dict["Adjusted Rand"],
        metrics_dict["NMI"],
        metrics_dict["Homogeneity"],
        metrics_dict["Completeness"],
        metrics_dict["V‑Measure"]
    )

    return plot_path, metrics_df

