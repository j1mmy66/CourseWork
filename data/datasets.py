from sklearn.datasets import make_blobs, make_moons, make_circles, load_digits
from sklearn.decomposition import PCA

from data.db import get_mnist_data


def load_blobs():
    X, y = make_blobs(n_samples=300, centers=3, random_state=42)
    return X, y

def load_moons():
    X, y = make_moons(n_samples=300, noise=0.05, random_state=42)
    return X, y

def load_circles():
    X, y = make_circles(n_samples=300, noise=0.05, factor=0.5, random_state=42)
    return X, y

def load_digit():
    X, y = load_digits()
    return X, y


def load_mnist_from_db():

    X, y = get_mnist_data()

    if X.shape[1] != 2:
        pca = PCA(n_components=2)
        X_reduced = pca.fit_transform(X)
    else:
        X_reduced = X
    return X_reduced, y