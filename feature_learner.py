import numpy as np
from abc import ABCMeta, abstractmethod

class IFeatureLearner:
    __metaclass__ = ABCMeta

    @classmethod 
    def __init__(self, final_centroids):
        self.final_centroids = final_centroids

    @classmethod
    def version(self): return "1.0"

    @abstractmethod
    def __feature_learner__(self, receptive_field): raise NotImplementedError

    @classmethod
    def __call__(self): return self.__feature_learner__

class KMeansSoft(IFeatureLearner):

    def __init__(self, final_centroids):
        super().__init__(final_centroids)
        self.final_centroids = final_centroids
        

    def __feature_learner__(self, receptive_field):
        z = [np.linalg.norm(np.subtract(receptive_field, centroid)) for centroid in self.final_centroids]
        mean_z = np.mean(z)
        return np.asarray([max(0, mean_z - z_k) for z_k in z]).astype("float")
    

class KMeansHard(IFeatureLearner):

    def __init__(self, final_centroids):
        super().__init__(final_centroids)
        self.final_centroids = final_centroids

    def __feature_learner__(self, receptive_field):
        z = [(k, np.linalg.norm(np.subtract(receptive_field, centroid))) for k, centroid in enumerate(self.final_centroids)]
        fk = [0 for k in range(len(self.final_centroids))]
        fk[min(z, key = lambda dist: dist[1])[0]] = 1
        return fk

kernel = {
    "kmeanssoft": KMeansSoft,
    "kmeanshard": KMeansHard
}