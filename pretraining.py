import time
from dynamic_configuration import algorithm2
from ab_llloyds import algorithm1
from progress_bar import printProgressBar
import numpy as np
from math import floor, ceil
import random

def dynamic_configure(V, d, k, m):
    alfa_h = 20
    epsilon = 1e-1

    n = len(V)
    m = min(n // k ** 2 if n > k ** 2 else 1, m) 
    r = n // m

    x = {i for i in range(n)}

    xxs = []
    for i in range(m):
        xs = set(random.sample(x, r))
        x -= xs
        xxs.append(xs)
    
    alfa_breakpoints = set()

    print ("Pre-search faze...")
    start_time = time.time()
    
    print("dynamic configuration beginning...")
    for i in range(m):
        Z = np.random.uniform(0.0, 1.0, k)
        alfa_interval_generator = algorithm2(V = V[np.asarray(list(xxs[i]))], d = d, k = k, Z = Z, alfa_h = alfa_h, epsilon = epsilon)

        for _, alfa_interval in alfa_interval_generator:
            alfa_breakpoints |= set(alfa_interval)

    best_score = 0
    best_alfa = 0

    j = 0
    print ("Computing best alfa parameter...")
    for alfa in alfa_breakpoints:
        scoreCH = [0 for _ in range(m)]
        for i in range(m):
            centroids, voronoi_tiling, _ = algorithm1(V = V[np.asarray(list(xxs[i]))], d = d, k = k, alfa = alfa,beta = 2, sum_of_squared_distances = False)
            centroid = np.mean(V[np.asarray(list(xxs[i]))], axis = 0)
            traceW = sum([sum([np.linalg.norm(np.subtract(instance, centroids[i]), d) for instance in voronoi_tiling[i]]) for i in range(k)])
            traceB = sum([len(voronoi_tiling[i]) * np.linalg.norm(np.subtract(centroid, centroids[i]), d) for i in range(k)])
            scoreCH[i] = (traceB / (k - 1)) / (traceW / (r - k))
        average_score = np.array(scoreCH).mean()
        
        if average_score > best_score:
            best_score = average_score
            best_alfa = alfa
     
        j += 1

    print ("Timp: (s)", time.time() - start_time)
    return best_alfa

def extract_centroids(V, d, k, alpha, beta):
    return algorithm1(V, d, k, alpha, beta, verbrose = True, sum_of_squared_distances = True if beta != 2 else False)
    
def kmeanscost(instances, centroids, voronoi_tiling, d, beta):
    k = len(centroids)
    final_sum = 0
    for i in range(k):
        dist_sum = 0
        for x in voronoi_tiling[i]:
            dist = np.linalg.norm(np.subtract(x, centroids[i]), d)
            dist_sum += dist ** 2
        final_sum += dist_sum

    return final_sum ** (1/2)

def chfitness(instances, centroids, voronoi_tiling, d, beta):
    k = len(centroids)
    n = len(instances)
    centroid = np.mean(instances, axis = 0)
    traceW = sum([sum([np.linalg.norm(np.subtract(voronoi_tiling[i][j], centroids[i]), d) ** 2 for j in range(len(voronoi_tiling[i]))]) for i in range(k)])
    traceB = sum([len(voronoi_tiling[i]) * np.linalg.norm(np.subtract(centroids[i], centroid)) for i in range(k)])
    
    return (traceB / (k - 1)) / (traceW / (n - k))


def performance_test(V, d, k, cf):
    alfa_min = 0
    alfa_max = 20
    beta_min = 1
    beta_max = 10

    alfa_step = (alfa_max - alfa_min) / 20
    beta_step = (beta_max - beta_min) / 10

    return_cost = []

    for alfa in np.arange(alfa_min, alfa_max + alfa_step, alfa_step):
        for beta in np.arange(beta_min, beta_max + beta_step, beta_step):
            alfa = ceil(alfa * 100) / 100
            beta = ceil(beta * 100) / 100
            print (alfa, beta)
            centroids, voronoi_tiling, _ = algorithm1(V, d, k, alfa, beta, verbrose = True, sum_of_squared_distances = True)
            cost = cost_function[cf](V, centroids, voronoi_tiling, d, beta)
            return_cost.append((alfa, beta, cost))

    return return_cost

cost_function = {
    'ch': chfitness,
    'kmeans': kmeanscost
}