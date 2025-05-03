import gradio as gr
import pandas as pd

from data.db import fetch_history_df


def build_history_page():
    with gr.TabItem("History"):
        # Пустой DataFrame при инициализации
        history_table = gr.DataFrame(
            value=pd.DataFrame(columns=[
                "id", "dataset", "algorithm", "silhouette", "davies_bouldin",
                "calinski_harabasz", "adjusted_rand", "nmi",
                "homogeneity", "completeness", "v_measure"
            ]),
            label="История кластеризаций"
        )
        refresh_btn = gr.Button("Обновить")
        # По клику обновляем таблицу
        refresh_btn.click(fn=fetch_history_df, outputs=history_table)