import gradio as gr

from interface.describtions.algorithm_describtion import ALGO_INFO_MD
from interface.describtions.dataset_describtion import DATASET_INFO_MD
from interface.describtions.generator_describtion import GENERATOR_INFO_MD
from interface.describtions.metrics_describtion import METRIC_INFO_MD
from interface.describtions.pca_description import PCA_INFO_MD
from interface.func import datasets_funcs, apply_clustering_or_generate, save_apply_clustering_or_generate


def build_sklearn_page():
    with gr.TabItem("sklearn") as sklearn_page:
        with gr.Group(elem_classes="custom-card"):
            gr.Markdown("## Кластеризация с использованием sklearn", elem_classes="card-title")

            # выбор режима: готовый датасет или генератор
            mode_selector_sklearn = gr.Radio(
                ["Выбор датасета", "Сгенерировать данные"],
                value="Выбор датасета",
                label="Источник данных",
                elem_classes="custom-radio"
            )

            # готовый датасет
            with gr.Accordion("ℹ️ Описание датасетов", open=False, elem_id="data-help") as dataset_info_acc:
                gr.Markdown(DATASET_INFO_MD)

            with gr.Accordion("ℹ️ Описание генератора", open=False, elem_id="generator-help",
                              visible=False) as generator_info_acc:
                gr.Markdown(GENERATOR_INFO_MD)

            dataset_dropdown_sklearn = gr.Dropdown(
                label="Выберите датасет",
                choices=list(datasets_funcs.keys()),
                value="Blobs",
                elem_classes="custom-dropdown"
            )

            # слайдеры генератора (по умолчанию скрыты)
            N_slider_sk = gr.Slider(1, 1000, value=300, step=1, label="N", visible=False)
            V_slider_sk = gr.Slider(1, 100, value=2, step=1, label="V", visible=False)
            K_slider_sk = gr.Slider(1, 10, value=3, step=1, label="K*", visible=False)
            nmin_slider_sk = gr.Slider(1, 50, value=5, step=1, label="n_min", visible=False)
            alpha_slider_sk = gr.Slider(0.01, 0.99, value=0.5, step=0.01, label="alpha", visible=False)

            # выбор алгоритма и кнопки
            with gr.Accordion("ℹ️ Описание алгоритмов", open=False, elem_id="algo-help"):
                gr.Markdown(ALGO_INFO_MD)
            algorithm_dropdown_sk = gr.Dropdown(
                label="Выберите алгоритм",
                choices=[
                    "KMeans", "DBSCAN", "AgglomerativeClustering",
                    "GaussianMixture", "SpectralClustering", "MeanShift"
                ],
                value="KMeans",
                elem_classes="custom-dropdown"
            )
            with gr.Accordion(label = "ℹ️ Максимальное время работы алгоритма", open=False):
                timeout_slider = gr.Slider(minimum=10, maximum=300, step=10, value=60, label="Тайм-аут (секунд)")

            run_button_sk = gr.Button("Кластеризовать", elem_classes="hover-button")

            with gr.Accordion("ℹ️ Отрисовка графиков", open=False, elem_id="metrics-help"):
                gr.Markdown(PCA_INFO_MD)

            output_image_sk = gr.Image(label="Результат", type="filepath", elem_classes="custom-image")

            with gr.Accordion("ℹ️ Что означают метрики?", open=False, elem_id="metrics-help"):
                gr.Markdown(METRIC_INFO_MD)

            metrics_dataframe_sk = gr.Dataframe(
                headers=["Метрика", "Значение"],
                datatype=["str", "number"],
                label="Метрики кластеризации",
                interactive=False,
                elem_classes="metrics-table"
            )


            # переключение видимости
            def update_visibility_sklearn(mode):
                show_gen = (mode == "Сгенерировать данные")
                return (
                    gr.update(visible=not show_gen),  # dataset_info
                    gr.update(visible=not show_gen),  # dataset
                    gr.update(visible=show_gen),  # gen_info
                    gr.update(visible=show_gen),  # N
                    gr.update(visible=show_gen),  # V
                    gr.update(visible=show_gen),  # K*
                    gr.update(visible=show_gen),  # n_min
                    gr.update(visible=show_gen),  # alpha
                )


            mode_selector_sklearn.change(
                fn=update_visibility_sklearn,
                inputs=[mode_selector_sklearn],
                outputs=[dataset_info_acc,
                         dataset_dropdown_sklearn,
                         generator_info_acc,
                         N_slider_sk,
                         V_slider_sk,
                         K_slider_sk,
                         nmin_slider_sk,
                         alpha_slider_sk
                         ]
            )

            # привязка кнопок
            run_button_sk.click(
                fn=save_apply_clustering_or_generate,
                inputs=[
                    mode_selector_sklearn,
                    dataset_dropdown_sklearn,
                    N_slider_sk, V_slider_sk, K_slider_sk,
                    nmin_slider_sk, alpha_slider_sk,
                    algorithm_dropdown_sk,
                    timeout_slider
                ],
                outputs=[output_image_sk, metrics_dataframe_sk]
            )