import numpy as np
from scipy.sparse import csr_matrix
from sklearn.cluster import KMeans
from scipy.sparse import diags

from algorithms.USPEC.uspec import uspec


def usenc(fea, k, M=20, distance='euclidean', p=1000, Knn=5, bcsLowK=20, bcsUpK=60):
    if bcsUpK < bcsLowK:
        bcsUpK = bcsLowK

    print("Generating an ensemble of", M, "base clusterings...")
    base_cls = usenc_ensemble_generation(fea, M, distance, p, Knn, bcsLowK, bcsUpK)

    print("Performing the consensus function...")
    labels = usenc_consensus_function(base_cls, k)

    return labels

def usenc_ensemble_generation(fea, M, distance='euclidean', p=1000, Knn=5, lowK=5, upK=15):
    N = fea.shape[0]
    if p > N:
        p = N

    members = np.zeros((N, M), dtype=int)


    for i in range(M):
        print(f"Generating the {i + 1}-th base clustering by U-SPEC...")
        Ks = np.random.randint(lowK, upK + 1, size=M)
        Ks = len(np.unique(Ks))
        members[:, i] = uspec(fea, Ks, distance, p, Knn)

    return members

def usenc_consensus_function(base_cls, k, max_tcut_km_iters=100, cnt_tcut_km_reps=3):
    N, M = base_cls.shape

    max_cls = np.max(base_cls, axis=0)
    for i in range(1, len(max_cls)):
        max_cls[i] += max_cls[i - 1]

    base_cls[:, 1:] += max_cls[:-1]

    B = csr_matrix((np.ones(N * M), (np.repeat(np.arange(N), M), base_cls.ravel())), shape=(N, max_cls[-1] + 1))
    col_sum = np.array(B.sum(axis=0)).flatten()
    B = B[:, col_sum > 0]

    labels = tcut_for_bipartite_graph(B, k, max_tcut_km_iters, cnt_tcut_km_reps)

    return labels

def tcut_for_bipartite_graph(B, Nseg, max_km_iters=100, cnt_reps=3):
    Nx, Ny = B.shape
    if Ny < Nseg:
        raise ValueError("Need more columns!")

    dx = np.array(B.sum(axis=1)).flatten()
    dx[dx == 0] = 1e-10
    #Dx = csr_matrix(np.diag(1.0 / dx))
    Dx = diags(1.0 / dx)
    Wy = B.T @ Dx @ B

    d = np.array(Wy.sum(axis=1)).flatten()
    #D = csr_matrix(np.diag(1.0 / np.sqrt(d)))
    D = diags(1.0 / np.sqrt(d))
    nWy = D @ Wy @ D

    eigvals, eigvecs = np.linalg.eigh(((nWy + nWy.T) / 2).toarray())
    Ncut_evec = D @ eigvecs[:, -Nseg:]

    evec = Dx @ B @ Ncut_evec
    evec = evec / (np.linalg.norm(evec, axis=1, keepdims=True) + 1e-10)

    kmeans = KMeans(n_clusters=Nseg, max_iter=max_km_iters, n_init=cnt_reps)
    labels = kmeans.fit_predict(evec)

    return labels