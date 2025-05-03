import gradio as gr

from interface.describtions.algorithm_describtion import ALGO_INFO_MD
from interface.css import css
from interface.describtions.dataset_describtion import DATASET_INFO_MD
from interface.describtions.generator_describtion import GENERATOR_INFO_MD
from interface.func import datasets_funcs, apply_clustering_or_generate
from interface.describtions.metrics_describtion import METRIC_INFO_MD
from interface.pages.sklearn_page import  build_sklearn_page
from interface.pages.uspec_page import  build_uspec_page

with gr.Blocks(css=css, theme=gr.themes.Soft(
        font=["Arial", "sans-serif"],  # основной шрифт
        font_mono=["Courier New", "monospace"]
)) as demo:
    with gr.Tabs(elem_classes="custom-tabs"):
        # ——— Склеарн-секция ———
        build_sklearn_page()

        # ——— USPEC/USENC-секция ———
        build_uspec_page()
        with gr.TabItem("2"):
            out1 = gr.Textbox(label="Результат")
        with gr.TabItem("3"):
            out1 = gr.Textbox(label="Результат")