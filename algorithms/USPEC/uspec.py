import numpy as np
from scipy.spatial.distance import cdist
from scipy.sparse import csr_matrix
from sklearn.cluster import KMeans

from algorithms.USPEC.pdist2_fast import pdist2_fast


def uspec(fea, Ks, distance='euclidean', p=1000, Knn=5, maxTcutKmIters=100, cntTcutKmReps=3):
    N = fea.shape[0]
    """number of objects"""
    p = min(p, N)
    """number of representatives"""

    # Get p representatives by hybrid selection
    RpFea = get_representatives_by_hybrid_selection(fea, p, distance)
    """set of p representatives"""


    # Approx. KNN
    cntRepCls = int(np.sqrt(p)) # считаем z' количество rep-cluster
    """z1 number of rep-cluster"""
    # pre-step 1
    # вычисляем rep-cluster
    kmeans = KMeans(n_clusters=cntRepCls, max_iter=20, random_state=0).fit(RpFea)
    repClsLabel = kmeans.labels_
    repClsCenters = kmeans.cluster_centers_
    """centers of rep-clusters"""





    centerDist = pdist2_fast(fea, repClsCenters, metric=distance) #расстояния от каждого центра rep-cluster до каждого из N
    """dist between center of each rep-cluster and each object in N"""

    #для каждого из N индекс ближайшего центра rep-cluster
    minCenterIdxs = np.argmin(centerDist, axis=1)
    """for each N ind of nearest rep-cluster"""

    #Для каждого  xi из N находим ближайщий элемент внутри ближайщего rep-cluster
    nearestRepInRpFeaIdx = np.zeros(N, dtype=int)
    """for each xi nearest elem in nearest rep-cluster"""
    for i in range(cntRepCls):
        mask = minCenterIdxs == i # объекты для которых этот rep-cluster ближайщий
        repSubset = RpFea[repClsLabel == i] # выбираем объекты из p которые пренадлежат этому rep-cluster
        nearestIdx = np.argmin(pdist2_fast(fea[mask], repSubset, metric=distance), axis=1) # находим индекс ближайшего объекта
        nearestRepInRpFeaIdx[mask] = np.flatnonzero(repClsLabel == i)[nearestIdx] # востанавливаем индексы

    # Для каждого p нашли его K` = 10K ближайщих соседей из p
    neighSize = 10 * Knn
    """K'"""
    RpFeaW = pdist2_fast(RpFea, RpFea, metric=distance)
    """paired dist between p"""
    RpFeaKnnIdx = np.argsort(RpFeaW, axis=1)[:, :neighSize + 1]
    """ind of K' neibours of each p"""

    RpFeaKnnDist = np.zeros((N, RpFeaKnnIdx.shape[1]))
    """dist between xi and K'"""
    for i in range(p):
        mask = nearestRepInRpFeaIdx == i # элементы из N для которых ближайший i представитель
        RpFeaKnnDist[mask] = cdist(fea[mask], RpFea[RpFeaKnnIdx[i]], metric=distance) # считаем расстояния от этим объектов до их потенциальных K' соседей

    RpFeaKnnIdxFull = RpFeaKnnIdx[nearestRepInRpFeaIdx]

    # Final KNN
    knnDist = np.zeros((N, Knn))
    knnIdx = np.zeros((N, Knn), dtype=int)
    for i in range(Knn):
        knnDist[:, i] = np.min(RpFeaKnnDist, axis=1)
        minIdx = np.argmin(RpFeaKnnDist, axis=1)
        knnIdx[:, i] = RpFeaKnnIdxFull[np.arange(N), minIdx]
        RpFeaKnnDist[np.arange(N), minIdx] = np.inf

    # Compute  B
    if distance == 'cosine':
        Gsdx = 1 - knnDist
    else:
        knnMeanDiff = np.mean(knnDist)
        Gsdx = np.exp(-(knnDist ** 2) / (2 * knnMeanDiff ** 2))

    Gsdx[Gsdx == 0] = np.finfo(float).eps
    Gidx = np.tile(np.arange(N), (Knn, 1)).T
    B = csr_matrix((Gsdx.ravel(), (Gidx.ravel(), knnIdx.ravel())), shape=(N, p))



    return tcut_for_bipartite_graph(B, Ks, maxTcutKmIters, cntTcutKmReps)


def get_representatives_by_hybrid_selection(fea, pSize, distance, cntTimes=10):
    N = fea.shape[0]
    bigPSize = min(cntTimes * pSize, N) # считаем p'
    bigRpFea = fea[np.random.choice(N, bigPSize, replace=False)] # выбираем p' оъектов
    kmeans = KMeans(n_clusters=pSize, max_iter=10, random_state=0).fit(bigRpFea) # кластеризуем
    return kmeans.cluster_centers_ # возвращаем p объектов - центры кластеров


def tcut_for_bipartite_graph(B, Nseg, maxKmIters, cntReps):

    Nx, Ny = B.shape

    if Ny < Nseg:
        raise ValueError("Need more columns!")


    dx = np.array(B.sum(axis=1)).ravel() # степени вершин
    Dx = np.diag(1.0 / np.maximum(dx, 1e-10)) # строим диагональную матрицу
    Wy = B.T @ Dx @ B

    d = np.array(Wy.sum(axis=1)).flatten()
    D = np.diag(1.0 / np.sqrt(d))
    nWy = D @ Wy @ D

    if hasattr(nWy, "toarray"):
        nWy = nWy.toarray()
    eigvals, eigvecs = np.linalg.eigh(nWy) # находим собственные значения и вектора
    idx = np.argsort(-eigvals)[:Nseg] # выбираем наибольшие собственные значения
    Ncut_evec = D @ eigvecs[:, idx] # нормализация собственных векторов
    evec = Dx @ B @ Ncut_evec
    evec /= np.linalg.norm(evec, axis=1, keepdims=True) + 1e-10

    kmeans = KMeans(n_clusters=Nseg, max_iter=maxKmIters, n_init=cntReps, random_state=0).fit(evec)
    return kmeans.labels_

