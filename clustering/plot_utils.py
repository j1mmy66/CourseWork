
import matplotlib
from sklearn.decomposition import PCA

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import os
import tempfile


def save_cluster_plot(X, labels, algorithm, dataset_name, filename="cluster_plot.png"):

    if X.shape[1] != 2:
        pca = PCA(n_components=2)
        X = pca.fit_transform(X)


    plt.figure(figsize=(6, 4))
    plt.scatter(X[:, 0], X[:, 1], c=labels, cmap="viridis", s=30)
    plt.xlabel('Признак 1')
    plt.ylabel('Признак 2')

    unique_labels = set(labels)
    summary_text = f"Алгоритм: {algorithm}\nДатасет: {dataset_name}\nКластеров: {len(unique_labels - {-1})}"

    plt.title(summary_text)



    # Сохранение графика во временную директорию
    img_path = os.path.join(tempfile.gettempdir(), filename)
    plt.savefig(img_path)
    plt.close()
    return img_path
