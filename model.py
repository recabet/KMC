import numpy as np
import pandas as pd
from matplotlib import pyplot as plt


class my_KMC:
    def __init__ (self, k: int = 3):
        """
        Initialize the K-means clustering model.

        :param k: Number of clusters.
        """
        self.k = k
        self.centr = None
        self.cluster_assignments = None
    
    @staticmethod
    def euclidean_distance (x, y) -> np.array:
        """
        Compute the Euclidean distance between points x and y.

        :param x: Array of points.
        :param y: Array of points.
        :return: Array of distances.
        """
        return np.linalg.norm(x - y, axis=1)
    
    def fit (self, X, iterations=100, limit=0.001) -> (np.array, np.array):
        """
        Fit the model to the data using the K-means algorithm.

        :param X: Input data.
        :param iterations: Maximum number of iterations.
        :param limit: Convergence threshold.
        :return: Centroids and cluster assignments.
        """
        if not isinstance(X, (np.ndarray, pd.DataFrame)):
            raise ValueError("Input data must be a numpy array or pandas DataFrame.")
        
        X = np.asarray(X)
        self.centr = np.random.uniform(low=np.amin(X, axis=0), high=np.amax(X, axis=0), size=(self.k, X.shape[1]))
        
        for _ in range(iterations):
            distances = np.array([self.euclidean_distance(X, centroid) for centroid in self.centr])
            cluster_assignments = np.argmin(distances, axis=0)
            
            new_centroids = np.array([
                X[cluster_assignments == i].mean(axis=0) if len(X[cluster_assignments == i]) > 0 else self.centr[i]
                for i in range(self.k)])
            
            if np.max(np.linalg.norm(self.centr - new_centroids, axis=1)) < limit:
                break
            
            self.centr = new_centroids
        
        self.cluster_assignments = cluster_assignments
        return self.centr, self.cluster_assignments
    
    def predict (self, X) -> np.array:
        """
        Predict cluster assignments for new data points.

        :param X: Input data.
        :return: Cluster assignments.
        """
        X = np.asarray(X)
        distances = np.array([self.euclidean_distance(X, centroid) for centroid in self.centr])
        return np.argmin(distances, axis=0)
    
    def mse (self, X) -> float:
        """
        Compute the mean squared error (WCSS).

        :param X: Input data.
        :return: Mean squared error.
        """
        X = np.asarray(X)
        wcss = sum(np.sum(np.linalg.norm(X[self.cluster_assignments == i] - self.centr[i], axis=1) ** 2) for i in
                   range(self.k))
        return wcss
    
    def silhouette_score (self, X) -> float:
        """
        Compute the silhouette score of the clustering.

        :param X: Input data.
        :return: Silhouette score.
        """
        if self.cluster_assignments is None:
            raise ValueError("The model must be fit before calculating the silhouette score.")
        
        X = np.asarray(X)
        silhouette_scores = np.zeros(len(X))
        
        for i in range(len(X)):
            same_cluster = X[self.cluster_assignments == self.cluster_assignments[i]]
            other_clusters = [X[self.cluster_assignments == j] for j in range(self.k) if
                              j != self.cluster_assignments[i]]
            
            a = np.mean(self.euclidean_distance(same_cluster, X[i])) if len(same_cluster) > 1 else 0
            b = np.min(
                [np.mean(self.euclidean_distance(cluster, X[i])) for cluster in other_clusters if len(cluster) > 0])
            
            silhouette_scores[i] = (b - a) / max(a, b) if max(a, b) > 0 else 0
        
        return float(np.mean(silhouette_scores))
    
    def elbow (self, X, max_k: int = 10, plot: bool = True) -> int:
        """
        Determine the optimal number of clusters using the elbow method.

        :param X: Input data.
        :param max_k: Maximum number of clusters to evaluate.
        :param plot: Whether to plot the results.
        :return: Optimal number of clusters.
        """
        if self.k <= 0:
            raise ValueError("The model is incorrectly set up (k must be non-negative).")
        
        error_history = []
        for i in range(1, max_k + 1):
            self.k = i
            self.fit(X)
            error_history.append(self.mse(X))
        
        first_order_diff = np.diff(error_history)
        second_order_diff = np.diff(first_order_diff)
        optimal_k = np.argmax(second_order_diff) + 2
        
        if plot:
            plt.plot(range(1, max_k + 1), error_history, marker='o', linestyle='-')
            plt.axvline(x=optimal_k, color='r', linestyle='--', label=f"Optimal k = {optimal_k}")
            plt.xlabel("Number of Clusters (k)")
            plt.ylabel("Mean Squared Error")
            plt.title("Elbow Method For Optimal k")
            plt.legend()
            plt.grid(True)
            plt.show()
        
        return optimal_k

