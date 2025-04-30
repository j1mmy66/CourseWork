import gradio as gr

from app.css import css
from interface.func import datasets_funcs, apply_clustering_or_generate

with gr.Blocks(css=css, theme=gr.themes.Soft(
        font=["Arial", "sans-serif"],  # основной шрифт
        font_mono=["Courier New", "monospace"]
)) as demo:
    with gr.Tabs(elem_classes="custom-tabs"):
        # ——— Склеарн-секция ———
        with gr.TabItem("sklearn"):
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
                algorithm_dropdown_sk = gr.Dropdown(
                    label="Выберите алгоритм",
                    choices=[
                        "KMeans", "DBSCAN", "AgglomerativeClustering",
                        "GaussianMixture", "SpectralClustering", "MeanShift"
                    ],
                    value="KMeans",
                    elem_classes="custom-dropdown"
                )
                run_button_sk = gr.Button("Кластеризовать", elem_classes="hover-button")
                output_image_sk = gr.Image(label="Результат", type="filepath", elem_classes="custom-image")


                # переключение видимости
                def update_visibility_sklearn(mode):
                    show_gen = (mode == "Сгенерировать данные")
                    return (
                        gr.update(visible=not show_gen),  # dataset
                        gr.update(visible=show_gen),  # N
                        gr.update(visible=show_gen),  # V
                        gr.update(visible=show_gen),  # K*
                        gr.update(visible=show_gen),  # n_min
                        gr.update(visible=show_gen),  # alpha
                    )


                mode_selector_sklearn.change(
                    fn=update_visibility_sklearn,
                    inputs=[mode_selector_sklearn],
                    outputs=[
                        dataset_dropdown_sklearn,
                        N_slider_sk, V_slider_sk, K_slider_sk,
                        nmin_slider_sk, alpha_slider_sk
                    ]
                )

                # привязка кнопок
                run_button_sk.click(
                    fn=apply_clustering_or_generate,
                    inputs=[
                        mode_selector_sklearn,
                        dataset_dropdown_sklearn,
                        N_slider_sk, V_slider_sk, K_slider_sk,
                        nmin_slider_sk, alpha_slider_sk,
                        algorithm_dropdown_sk
                    ],
                    outputs=output_image_sk
                )

        # ——— USPEC/USENC-секция ———
        with gr.TabItem("USENC/USPEC"):
            with gr.Group(elem_classes="custom-card"):
                gr.Markdown("## Кластеризация USPEC/USENC", elem_classes="card-title")

                # выбор режима: датасет или генерация
                mode_selector = gr.Radio(
                    ["Выбор датасета", "Сгенерировать данные"],
                    value="Выбор датасета",
                    label="Источник данных",
                    elem_classes="custom-radio"
                )

                # контейнер для готового датасета
                dataset_dropdown2 = gr.Dropdown(
                    label="Выберите датасет",
                    choices=list(datasets_funcs.keys()),
                    value="Blobs",
                    elem_classes="custom-dropdown"
                )

                # слайдеры для генератора (по умолчанию скрыты)
                N_slider = gr.Slider(1, 1000, value=300, step=1, label="N", visible=False)
                V_slider = gr.Slider(1, 100, value=2, step=1, label="V", visible=False)
                K_slider = gr.Slider(1, 10, value=3, step=1, label="K*", visible=False)
                nmin_slider = gr.Slider(1, 50, value=5, step=1, label="n_min", visible=False)
                alpha_slider = gr.Slider(0.01, 0.99, value=0.5, step=0.01, label="alpha", visible=False)

                # выбор алгоритма и кнопка
                algorithm_dropdown2 = gr.Dropdown(
                    label="Выберите алгоритм",
                    choices=["USPEC", "USENC"],
                    value="USPEC",
                    elem_classes="custom-dropdown"
                )
                run_button2 = gr.Button("Кластеризовать", elem_classes="hover-button")
                output_image2 = gr.Image(label="Результат", type="filepath", elem_classes="custom-image")


                # при смене режима переключаем видимость
                def update_visibility(mode):
                    return (
                        gr.update(visible=(mode == "Выбор датасета")),  # датасет
                        gr.update(visible=(mode == "Сгенерировать данные")),  # N
                        gr.update(visible=(mode == "Сгенерировать данные")),  # V
                        gr.update(visible=(mode == "Сгенерировать данные")),  # K*
                        gr.update(visible=(mode == "Сгенерировать данные")),  # n_min
                        gr.update(visible=(mode == "Сгенерировать данные")),  # alpha
                    )


                mode_selector.change(
                    fn=update_visibility,
                    inputs=[mode_selector],
                    outputs=[dataset_dropdown2, N_slider, V_slider, K_slider, nmin_slider, alpha_slider]
                )

                # единый callback на кнопку
                run_button2.click(
                    fn=apply_clustering_or_generate,
                    inputs=[
                        mode_selector,
                        dataset_dropdown2,
                        N_slider, V_slider, K_slider, nmin_slider, alpha_slider,
                        algorithm_dropdown2
                    ],
                    outputs=output_image2
                )
        with gr.TabItem("2"):
            out1 = gr.Textbox(label="Результат")
        with gr.TabItem("3"):
            out1 = gr.Textbox(label="Результат")