# app/db.py
import os

import numpy as np
import psycopg2
from psycopg2.extras import execute_values

def get_connection():
    connection = psycopg2.connect(
        host=os.environ.get("DB_HOST", "localhost"),
        port=os.environ.get("DB_PORT", 5432),
        database=os.environ.get("DB_NAME", "history"),
        user=os.environ.get("DB_USER", "postgres"),
        password=os.environ.get("DB_PASSWORD", "postgres")
    )
    return connection

def create_history_table():
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("""
    CREATE TABLE IF NOT EXISTS history_data (
        id SERIAL PRIMARY KEY,
        dataset TEXT,
        algorithm TEXT,
        silhouette TEXT,
        davies_bouldin TEXT,
        calinski_harabasz TEXT, 
        adjusted_rand TEXT,
        nmi TEXT,
        homogeneity TEXT,
        completeness TEXT,
        v_measure TEXT
    );
    """)
    conn.commit()
    cur.close()
    conn.close()

def insert_history_data(dataset, algorithm, silhouette, davies_bouldin, calinski_harabasz,
                        adjusted_rand, nmi, homogeneity, completeness, v_measure):

    conn = get_connection()
    cur = conn.cursor()

    query = """
        INSERT INTO history_data (
            dataset, algorithm, silhouette, davies_bouldin, calinski_harabasz,
            adjusted_rand, nmi, homogeneity, completeness, v_measure
        )
        VALUES %s
    """

    # Собираем данные в виде списка кортежей (execute_values требует именно такой формат)
    values = [(
        dataset, algorithm, silhouette, davies_bouldin, calinski_harabasz,
        adjusted_rand, nmi, homogeneity, completeness, v_measure
    )]

    execute_values(cur, query, values)

    conn.commit()
    cur.close()
    conn.close()

import pandas as pd
from psycopg2 import sql

def fetch_history_df(limit: int = None) -> pd.DataFrame:
    """
    Извлекает таблицу history_data в pandas.DataFrame.
    """
    conn = get_connection()
    # Базовый SQL
    query = "SELECT * FROM history_data"
    if limit is not None:
        query += f" LIMIT {limit}"
    # pandas сама откроет и закроет курсор
    df = pd.read_sql_query(query, conn)
    conn.close()
    return df
