# scripts/seed_db.py
import tensorflow as tf
from data.db import create_mnist_table, insert_mnist_data


def load_mnist_data():
    (x_train, y_train), _ = tf.keras.datasets.mnist.load_data()

    return x_train, y_train


def seed_database():
    print("Создаём таблицу MNIST (если ещё не существует)...")
    create_mnist_table()
    print("Таблица готова.")

    x_train, y_train = load_mnist_data()


    num_samples = 1000
    data = []
    for i in range(num_samples):

        image_bytes = x_train[i].tobytes()
        label = int(y_train[i])
        data.append((image_bytes, label))

    print(f"Вставляем {num_samples} образцов в базу данных...")
    insert_mnist_data(data)
    print("Заполнение базы данных завершено.")


if __name__ == '__main__':
    seed_database()
