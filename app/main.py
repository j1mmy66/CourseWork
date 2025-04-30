
import time
import gradio as gr

from app.css import css
from app.datasets import load_blobs, load_moons, load_circles, load_iris, load_mnist_from_db, load_iriss
from app.clustering_sklearn import get_default_clusters, perform_clustering, compute_silhouette
from app.plot_utils import save_cluster_plot


datasets_funcs = {
    "Blobs": load_blobs,
    "Moons": load_moons,
    "Circles": load_circles,
    "Iris": load_iriss,
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

with gr.Blocks(css=css, theme=gr.themes.Soft(
        font=["Arial", "sans-serif"],        # основной шрифт
        font_mono=["Courier New", "monospace"]
)) as demo:
    with gr.Tabs(elem_classes="custom-tabs"):
        # ——— Склеарн-секция ———
        with gr.TabItem("sklearn"):
            with gr.Group(elem_classes="custom-card"):
                gr.Markdown("## Кластеризация с использованием sklearn", elem_classes="card-title")
                with gr.Row():
                    dataset_dropdown = gr.Dropdown(
                        label="Выберите датасет",
                        choices=list(datasets_funcs.keys()),
                        value="Blobs",
                        elem_classes="custom-dropdown"
                    )
                    algorithm_dropdown = gr.Dropdown(
                        label="Выберите алгоритм",
                        choices=[
                            "KMeans", "DBSCAN", "AgglomerativeClustering",
                            "GaussianMixture", "SpectralClustering", "MeanShift"
                        ],
                        value="KMeans",
                        elem_classes="custom-dropdown"
                    )
                run_button = gr.Button("Кластеризовать", elem_classes="hover-button")
                stream_button = gr.Button("Показать поток", elem_classes="hover-button")
                output_image = gr.Image(label="Результат", type="filepath", elem_classes="custom-image")

                run_button.click(apply_clustering,
                                 inputs=[dataset_dropdown, algorithm_dropdown],
                                 outputs=output_image)
                stream_button.click(stream_clustering,
                                    inputs=[dataset_dropdown, algorithm_dropdown],
                                    outputs=output_image)

        # ——— USPEC/USENC-секция ———
        with gr.TabItem("USENC/USPEC"):
            with gr.Group(elem_classes="custom-card"):
                gr.Markdown("## Кластеризация USPEC/USENC", elem_classes="card-title")
                with gr.Row():
                    dataset_dropdown2 = gr.Dropdown(
                        label="Выберите датасет",
                        choices=list(datasets_funcs.keys()),
                        value="Blobs",
                        elem_classes="custom-dropdown"
                    )
                    algorithm_dropdown2 = gr.Dropdown(
                        label="Выберите алгоритм",
                        choices=["USPEC", "USENC"],
                        value="USPEC",
                        elem_classes="custom-dropdown"
                    )
                run_button2 = gr.Button("Кластеризовать", elem_classes="hover-button")
                output_image2 = gr.Image(label="Результат", type="filepath", elem_classes="custom-image")

                run_button2.click(apply_clustering,
                                  inputs=[dataset_dropdown2, algorithm_dropdown2],
                                  outputs=output_image2)

        with gr.TabItem("2"):
            out1 = gr.Textbox(label="Результат")
        with gr.TabItem("3"):
            out1 = gr.Textbox(label="Результат")
if __name__ == '__main__':

    demo.queue().launch(server_name="0.0.0.0", server_port=7860, share=True)

