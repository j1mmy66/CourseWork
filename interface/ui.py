import gradio as gr

from interface.describtions.algorithm_describtion import ALGO_INFO_MD
from interface.css import css
from interface.describtions.dataset_describtion import DATASET_INFO_MD
from interface.describtions.generator_describtion import GENERATOR_INFO_MD
from interface.func import datasets_funcs, apply_clustering_or_generate
from interface.describtions.metrics_describtion import METRIC_INFO_MD
from interface.pages.acmk_page import build_acmk_page
from interface.pages.history_page import build_history_page
from interface.pages.lwec_page import build_lwec_page
from interface.pages.sklearn_page import  build_sklearn_page
from interface.pages.uspec_page import  build_uspec_page

with gr.Blocks(css=css, theme=gr.themes.Soft(
        font=["Arial", "sans-serif"],  # основной шрифт
        font_mono=["Courier New", "monospace"]
)) as demo:
    with gr.Tabs(elem_classes="custom-tabs"):

        build_sklearn_page()

        build_uspec_page()

        build_lwec_page()

        build_acmk_page()

        build_history_page()