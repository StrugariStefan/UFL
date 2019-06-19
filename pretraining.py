import time
from dynamic_configuration import algorithm2
from ab_llloyds import algorithm1
from progress_bar import printProgressBar
import numpy as np
from math import floor
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
    # progress_bar.printProgressBar(0, m - 1, prefix = 'Progress:', suffix = 'Complete', length = 50)
    
    print("dynamic configuration beginning...")
    for i in range(m):
        # print ("i = " + str(i))
        Z = np.random.uniform(0.0, 1.0, k)
        alfa_interval_generator = algorithm2(V = V[np.asarray(list(xxs[i]))], d = d, k = k, Z = Z, alfa_h = alfa_h, epsilon = epsilon)

        for _, alfa_interval in alfa_interval_generator:
            alfa_breakpoints |= set(alfa_interval)
            
        # progress_bar.printProgressBar(i + 1, m - 1, prefix = 'Progress:', suffix = 'Complete', length = 50)

    print (alfa_breakpoints)

    best_score = 0
    best_alfa = 0

    j = 0
    print ("Computing best alfa parameter...")
    # progress_bar.printProgressBar(0, len(alfa_breakpoints), prefix = 'Progress:', suffix = 'Complete', length = 50)
    for alfa in alfa_breakpoints:
        # print ("j = " + str(j))
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

        # progress_bar.printProgressBar(j + 1, len(alfa_breakpoints), prefix = 'Progress:', suffix = 'Complete', length = 50)        
        j += 1

    print ("Timp: (s)", time.time() - start_time)
    return best_alfa

def extract_centroids(V, d, k, alpha, beta):
    return algorithm1(V, d, k, alpha, beta, verbrose = True, sum_of_squared_distances = True if beta != 2 else False)