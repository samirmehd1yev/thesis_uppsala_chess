from typing import List
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from models.data_classes import FeatureVector

class StyleClusterer:
    def __init__(self, n_clusters: int = 4):
        self.scaler = StandardScaler()
        self.kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        
    def fit(self, feature_vectors: List[FeatureVector]):
        """Fit clustering model"""
        X = np.vstack([fv.to_array() for fv in feature_vectors])
        X_scaled = self.scaler.fit_transform(X)
        self.kmeans.fit(X_scaled)
        
    def predict(self, feature_vector: FeatureVector) -> int:
        """Predict cluster for new game"""
        X = feature_vector.to_array().reshape(1, -1)
        X_scaled = self.scaler.transform(X)
        return int(self.kmeans.predict(X_scaled)[0])
