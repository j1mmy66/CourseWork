import numpy as np


def generate_synthetic_data(N, V, K_star, n_min, alpha):
    """
    Генерирует синтетический набор данных X размером N x V с K_star кластерами.

    Параметры:
    - N (int): общее число объектов (строк) в выборке.
    - V (int): число признаков (столбцов) у каждого объекта.
    - K_star (int): число генерируемых кластеров (K*).
    - n_min (int): минимальный размер каждого кластера (минимум объектов в кластере).
    - alpha (float): параметр степени интерференции кластеров (0 < alpha < 1).

    Возвращает:
    - X (np.ndarray): матрица размера N x V, содержащая сгенерированные объекты.
    - labels (np.ndarray): вектор длины N, где labels[i] = индекс кластера (0..K*-1) для i-го объекта.
    """
    # Проверка входных параметров
    if K_star * n_min > N:
        raise ValueError("N меньше, чем K_star * n_min – невозможно обеспечить минимум объектов в каждом кластере")
    if not (0 < alpha < 1):
        raise ValueError("alpha должен быть в диапозоне (0, 1).")

    # 1. Определяем размеры кластеров
    cluster_sizes = np.full(K_star, n_min, dtype=int)  # каждый кластер получает минимум n_min объектов
    remaining = N - n_min * K_star  # остаток объектов, которые нужно распределить
    if remaining > 0:
        # Случайно распределяем оставшиеся объекты по кластерам
        extra = np.random.multinomial(remaining, [1.0 / K_star] * K_star)
        cluster_sizes += extra

    # 2. Генерируем центры кластеров из U(alpha-1, 1-alpha)^V
    centers = np.random.uniform(alpha - 1, 1 - alpha, size=(K_star, V))

    # 3. Генерируем стандартные отклонения признаков для каждого кластера из [0.05, 0.10]
    stds = np.random.uniform(0.05, 0.10, size=(K_star, V))

    # 3 (продолжение). Генерируем объекты для каждого кластера
    X = np.empty((N, V))
    labels = np.empty(N, dtype=int)
    start = 0
    for k in range(K_star):
        n_k = cluster_sizes[k]
        # Генерируем n_k точек для кластера k
        # np.random.randn(n_k, V) создает массив n_k x V из N(0,1)
        # умножаем его на stds[k] (вектор std для кластера k) и прибавляем центр k-го кластера
        X_cluster = np.random.randn(n_k, V) * stds[k] + centers[k]
        X[start:start + n_k] = X_cluster
        labels[start:start + n_k] = k
        start += n_k

    return X, labels