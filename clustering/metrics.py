
from sklearn.metrics import (
    davies_bouldin_score,
    calinski_harabasz_score,
    adjusted_rand_score,
    normalized_mutual_info_score,
    homogeneity_completeness_v_measure,
    silhouette_score
)


def compute_silhouette(X, labels):
    unique_labels = set(labels)
    if len(unique_labels - {-1}) > 1:
        return silhouette_score(X, labels)
    return None

def compute_davies_bouldin(X, labels):
    unique_labels = set(labels)
    if len(unique_labels - {-1}) > 1:
        return davies_bouldin_score(X, labels)
    return None

# 2. Calinski–Harabasz Index
def compute_calinski_harabasz(X, labels):
    unique_labels = set(labels)
    if len(unique_labels - {-1}) > 1:
        return calinski_harabasz_score(X, labels)
    return None

# 3. Adjusted Rand Index (требуются истинные метки)
def compute_adjusted_rand(y_true, labels):
    return adjusted_rand_score(y_true, labels)

# 4. Normalized Mutual Information
def compute_nmi(y_true, labels):
    return normalized_mutual_info_score(y_true, labels)

# 5. Homogeneity, Completeness, V-Measure (возвращает все три)
def compute_hcv(y_true, labels):
    return homogeneity_completeness_v_measure(y_true, labels)