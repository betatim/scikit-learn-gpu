import time
import numpy as np
import cupy as cp

from sklearn.cluster import _kmeans

from pylibraft.distance import pairwise_distance as pw_distances


def pairwise_distances_argmin(X, Y):
    distances = cp.zeros((X.shape[0], Y.shape[0]), dtype=X.dtype)
    pw_distances(X, Y, distances)
    return distances.argmin(axis=1)


class KMeansEngine(_kmeans.KMeansCythonEngine):
    def __init__(self, estimator):
        # print("Rafting!")
        self.estimator = estimator

    def prepare_fit(self, X, y=None, sample_weight=None):
        X, y, sample_weight = super().prepare_fit(X, y, sample_weight)

        # Used later to unshift X again, need to explicity convert to
        # cupy array
        self.X_mean = cp.asarray(self.X_mean)

        return X, y, sample_weight

    def init_centroids(self, X):
        return cp.asarray(super().init_centroids(X))

    def get_labels(self, X, sample_weight):
        X = cp.asarray(X)
        labels = pairwise_distances_argmin(X, self.estimator.cluster_centers_)
        return labels

    def kmeans_single(self, X, sample_weight, current_centroids):
        X = cp.asarray(X)
        sample_weight = cp.asarray(sample_weight)

        labels = None
        centers = None
        for n_iter in range(self.estimator.max_iter):
            # E(xpectation)
            new_labels = pairwise_distances_argmin(
                X,
                current_centroids,
            )

            # M(aximization)
            # Compute the new centroids using the weighted sum of points in each cluster
            new_centers = cp.array(
                [
                    (
                        X[new_labels == i]
                        * sample_weight.reshape((-1, 1))[new_labels == i]
                    ).sum(0)
                    / sample_weight[new_labels == i].sum()
                    for i in range(self.estimator.n_clusters)
                ]
            )

            if n_iter > 0:
                if cp.array_equal(labels, new_labels):
                    break
                else:
                    center_shift = cp.power(centers - new_centers, 2).sum()
                    if center_shift <= self.tol:
                        break

            centers = new_centers
            labels = new_labels

        inertia = 0.0
        for n, center in enumerate(centers):
            inertia += (
                cp.power(X[labels == n] - center, 2).sum(1) * sample_weight[labels == n]
            ).sum()

        return labels, inertia, centers, n_iter + 1
