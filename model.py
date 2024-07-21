import numpy as np


class my_KMC:
    def __init__ (self, k: int = 3):
        self.k = k
        self.centr = None
        self.cluster_assignments = None
    
    @staticmethod
    def euclidean_distance (x, y):
        return np.linalg.norm(x - y, axis=1)
    
    def fit (self, X, iterations=100, limit=0.001):
        
        self.centr = np.random.uniform(low=np.amin(X, axis=0), high=np.amax(X, axis=0), size=(self.k, X.shape[1]))
        
        for _ in range(iterations):
          
            distances = np.array([self.euclidean_distance(X, centroid) for centroid in self.centr])
            cluster_assignments = np.argmin(distances, axis=0)
            
            new_centroids = np.array([
                X[cluster_assignments == i].mean(axis=0) if len(X[cluster_assignments == i]) > 0 else self.centr[i]
                for i in range(self.k) ])
            
            if np.max(np.linalg.norm(self.centr - new_centroids, axis=1)) < limit:
                break
            
            self.centr = new_centroids
            
        self.cluster_assignments = cluster_assignments
        
        return self.centr, cluster_assignments
    
    def predict (self, X):
        distances = np.array([self.euclidean_distance(X, centroid) for centroid in self.centr])
        cluster_assignments = np.argmin(distances, axis=0)
        return cluster_assignments
    
    def calculate_wcss (self, X):
        wcss = 0
        for i in range(self.k):
            cluster_points = X[self.cluster_assignments == i]
            centroid = self.centr[i]
            wcss += np.sum(np.linalg.norm(cluster_points - centroid, axis=1) ** 2)
        return wcss
    
    def silhouette_score (self, X):
        if self.cluster_assignments is None:
            raise ValueError("The model must be fit before calculating the silhouette score.")
        
        silhouette_scores = np.zeros(len(X))
        
        for i in range(len(X)):
            same_cluster = X[self.cluster_assignments == self.cluster_assignments[i]]
            other_clusters = [X[self.cluster_assignments == j] for j in range(self.k) if
                              j != self.cluster_assignments[i]]
            
            if len(same_cluster) > 1:
                a = np.mean(np.linalg.norm(same_cluster - X[i], axis=1))
            else:
                a = 0
                
            b = np.min([np.mean(np.linalg.norm(cluster - X[i], axis=1)) for cluster in other_clusters])
            silhouette_scores[i] = (b - a) / max(a, b)
        
        return np.mean(silhouette_scores)
