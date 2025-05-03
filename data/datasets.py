from sklearn.datasets import make_blobs, make_moons, make_circles, load_digits
from sklearn.decomposition import PCA




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
    digits = load_digits()
    X, y = digits.data, digits.target

    return X, y


