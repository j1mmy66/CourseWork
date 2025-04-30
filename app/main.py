
import time
import gradio as gr

from generator.generator import generate_synthetic_data
from app.css import css
from app.datasets import load_blobs, load_moons, load_circles,  load_mnist_from_db
from app.clustering_sklearn import get_default_clusters, perform_clustering, compute_silhouette
from app.plot_utils import save_cluster_plot
from interface.ui import demo







if __name__ == '__main__':

    demo.queue().launch(server_name="0.0.0.0", server_port=7860, share=True)

