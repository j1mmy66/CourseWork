from sklearn.datasets import make_blobs, make_moons, make_circles, load_iris
from sklearn.decomposition import PCA

from app.db import get_mnist_data


def load_blobs():
    X, y = make_blobs(n_samples=300, centers=3, random_state=42)
    return X, y

def load_moons():
    X, y = make_moons(n_samples=300, noise=0.05, random_state=42)
    return X, y

def load_circles():
    X, y = make_circles(n_samples=300, noise=0.05, factor=0.5, random_state=42)
    return X, y

def load_iris():
    iris = load_iris()
    X = iris.data[:, :2]
    y = iris.target
    return X, y

def load_mnist_from_db():

    X, y = get_mnist_data()

    if X.shape[1] != 2:
        pca = PCA(n_components=2)
        X_reduced = pca.fit_transform(X)
    else:
        X_reduced = X
    return X_reduced, y