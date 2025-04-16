
import time
import gradio as gr

from app.datasets import load_blobs, load_moons, load_circles, load_iris, load_mnist_from_db
from app.clustering import get_default_clusters, perform_clustering, compute_silhouette
from app.plot_utils import save_cluster_plot


datasets_funcs = {
    "Blobs": load_blobs,
    "Moons": load_moons,
    "Circles": load_circles,
    "Iris": load_iris,
    "MNIST": load_mnist_from_db
}


def apply_clustering(dataset_name, algorithm):
    X, _ = datasets_funcs[dataset_name]()
    n_clusters = get_default_clusters(dataset_name)
    labels, centers = perform_clustering(X, algorithm, n_clusters)
    score = compute_silhouette(X, labels)
    return save_cluster_plot(X, labels, algorithm, dataset_name, centers, score, "cluster_plot.png")


def stream_clustering(dataset_name, algorithm):
    X, _ = datasets_funcs[dataset_name]()
    n_clusters = get_default_clusters(dataset_name)
    labels, centers = perform_clustering(X, algorithm, n_clusters)
    score = compute_silhouette(X, labels)

    n_samples = X.shape[0]
    for i in range(1, n_samples + 1):
        img_path = save_cluster_plot(X[:i, :], labels[:i], algorithm, dataset_name, centers, score,
                                     "cluster_plot_stream.png")
        yield img_path
        time.sleep(0.05)


with gr.Blocks() as demo:
    gr.Markdown("## Кластеризация с использованием sklearn")
    with gr.Row():
        dataset_dropdown = gr.Dropdown(label="Выберите датасет", choices=list(datasets_funcs.keys()), value="Blobs")
        algorithm_dropdown = gr.Dropdown(label="Выберите алгоритм",
                                         choices=["KMeans", "DBSCAN", "AgglomerativeClustering",
                                                  "GaussianMixture", "SpectralClustering", "MeanShift"],
                                         value="KMeans")
    run_button = gr.Button("Кластеризовать")
    stream_button = gr.Button("Показать поток")
    output_image = gr.Image(label="Результат", type="filepath")

    run_button.click(apply_clustering, inputs=[dataset_dropdown, algorithm_dropdown], outputs=output_image)
    stream_button.click(fn=stream_clustering, inputs=[dataset_dropdown, algorithm_dropdown], outputs=output_image)

if __name__ == '__main__':

    demo.queue().launch(server_name="0.0.0.0", server_port=7860)

