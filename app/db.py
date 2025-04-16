# app/db.py
import os

import numpy as np
import psycopg2
from psycopg2.extras import execute_values

def get_connection():
    connection = psycopg2.connect(
        host=os.environ.get("DB_HOST", "localhost"),
        port=os.environ.get("DB_PORT", 5432),
        database=os.environ.get("DB_NAME", "mnist_db"),
        user=os.environ.get("DB_USER", "postgres"),
        password=os.environ.get("DB_PASSWORD", "postgres")
    )
    return connection

def create_mnist_table():
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("""
    CREATE TABLE IF NOT EXISTS mnist_data (
        id SERIAL PRIMARY KEY,
        image_data BYTEA,
        label INTEGER
    );
    """)
    conn.commit()
    cur.close()
    conn.close()

def insert_mnist_data(data):

    conn = get_connection()
    cur = conn.cursor()
    query = "INSERT INTO mnist_data (image_data, label) VALUES %s"
    execute_values(cur, query, data)
    conn.commit()
    cur.close()
    conn.close()

def get_mnist_data():

    conn = get_connection()
    cur = conn.cursor()
    cur.execute("SELECT image_data, label FROM mnist_data ORDER BY id ASC;")
    rows = cur.fetchall()
    cur.close()
    conn.close()

    images = []
    labels = []
    for row in rows:
        image_bytes, label = row

        image = np.frombuffer(image_bytes, dtype=np.uint8).reshape(28, 28)

        images.append(image.flatten())
        labels.append(label)
    X = np.array(images)
    y = np.array(labels)
    return X, y
