import gradio as gr

from interface.func import datasets_funcs, apply_clustering_or_generate


def build_uspec_page():
    with gr.TabItem("usenc/uspec") as uspec_page:
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
                outputs=[
                    dataset_dropdown2,
                    N_slider,
                    V_slider,
                    K_slider,
                    nmin_slider,
                    alpha_slider]
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