
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import os
import tempfile


def save_cluster_plot(X, labels, algorithm, dataset_name, centers=None, silhouette=None, filename="cluster_plot.png"):
    plt.figure(figsize=(6, 4))
    plt.scatter(X[:, 0], X[:, 1], c=labels, cmap="viridis", s=30)
    plt.xlabel('Признак 1')
    plt.ylabel('Признак 2')

    unique_labels = set(labels)
    summary_text = f"Алгоритм: {algorithm}\nДатасет: {dataset_name}\nКластеров: {len(unique_labels - {-1})}"
    if silhouette is not None:
        summary_text += f"\nSilhouette: {silhouette:.2f}"
    else:
        summary_text += "\nSilhouette: N/A"
    plt.title(summary_text)

    if centers is not None:
        plt.scatter(centers[:, 0], centers[:, 1], marker="x", c="red", s=100, linewidths=3, label="Центры")
        plt.legend()

    # Сохранение графика во временную директорию
    img_path = os.path.join(tempfile.gettempdir(), filename)
    plt.savefig(img_path)
    plt.close()
    return img_path
