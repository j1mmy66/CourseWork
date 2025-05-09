import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances
from sklearn.preprocessing import StandardScaler
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform



def run_multiple_kmeans(X, n_clusters, n_runs=10, max_iter=300, random_state=42):
  np.random.seed(random_state)
  N = X.shape[0]
  labels_array = np.zeros((n_runs, N), dtype=int)
  for i in range(n_runs):
    km = KMeans(n_clusters=n_clusters,
      init='k-means++',
      n_init=1,
      max_iter=max_iter,
    random_state=random_state + i)
    km.fit(X)

    labels_array[i] = km.labels_
  return labels_array

def build_coassociation_matrix(labels_array):
  n_runs, N = labels_array. shape
  M = np.zeros((N, N))
  for r in range(n_runs):
    for i in range(N):
      for j in range(i+1, N):
        if labels_array[r, i] == labels_array[r, j]:
          M[i, j] += 1
          M[j, i] += 1
  M /= n_runs
  np.fill_diagonal(M, 1.0)
  return M

def initial_consensus_labels(M, n_clusters):
  dist_vec = squareform(1.0 - M, checks=False)
  Z = linkage(dist_vec, method='average')
  labels = fcluster(Z, t=n_clusters, criterion='maxclust') - 1
  return labels

def refine_consensus_labels(M, labels, max_iter=10, tol=1e-4):
  labels = labels.copy()
  N = M.shape[0]
  for iteration in range(max_iter):
    old_labels = labels.copy()
    unique_clusters = np.unique(labels)
    cluster_indices = {uc: np.where(labels == uc)[0] for uc in unique_clusters}
    for i in range(N):
      current_cluster = labels[i]
      best_cluster = current_cluster
      best_score = -1.0
      for c in unique_clusters:
        inds = cluster_indices[c]
        if len(inds) == 0:
          continue
        score = np.mean(M[i, inds])
        if score > best_score:
          best_score = score
          best_cluster = c
      labels[i] = best_cluster
    changed = np.sum(old_labels != labels)
    changed_ratio = changed / float(N)
    if changed_ratio < tol:
      break
  return labels

def acmk_clustering(X, n_clusters, n_runs=10, max_iter_kmeans=300,
random_state=42, refine_max_iter=10, refine_tol=1e-4):
  labels_array = run_multiple_kmeans(X,
  n_clusters=n_clusters,
  n_runs=n_runs,
  max_iter=max_iter_kmeans,
  random_state=random_state)
  M = build_coassociation_matrix(labels_array)
  init_labels = initial_consensus_labels(M, n_clusters)
  final_labels = refine_consensus_labels(M, init_labels,
  max_iter=refine_max_iter,
  tol=refine_tol)

  return final_labels, M

