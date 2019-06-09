from d_sampling import d_alfa_sample
import time
import progress_bar
import numpy as np

def initialize(instances, d, k, alfa, sort_by_distance = False):
    Z = np.random.uniform(0.0, 1.0, k)
    C = []
    V = instances[:]
    original_indexes = []
    
    for t in range(k):
        intervals, V, C, instance_index = d_alfa_sample(C, V, d, alfa, Z[t], sort_by_distance)
        original_indexes.append(list(filter(lambda element: all(element[1] == V[instance_index]), enumerate(instances)))[0][0])
        V = np.delete(V, instance_index, axis = 0)

    return C, V, intervals, original_indexes, Z

def get_Voronoi_tiling(C, V, k, d, beta = 2):
    Voronoi_tiling = [[] for _ in range(k)]
    clustering_indexes = [0 for _ in range(len(V))]
    for instance_index, instance in enumerate(V):
        # centroid_proximal = min(C, key = lambda centroid: np.linalg.norm(np.subtract(instance, centroid), d) """** beta""", default = None)
        centroid_proximal = min(C, key = lambda centroid: np.linalg.norm(np.subtract(instance, centroid), d), default = None)        
        for i in range(len(C)):
            if np.array_equal(C[i], centroid_proximal):
                index = i
                break
        Voronoi_tiling[index].append(instance)
        clustering_indexes[instance_index] = index
    return Voronoi_tiling, clustering_indexes

def all_equal(C1, C2):
    for i in range(len(C1)):
        if np.array_equal(C1[i], C2[i]) == False:
            return False
    return True

def algorithm1(V, d, k, alfa, beta, initial_centroids = None, T_max = 3, verbrose = False, sum_of_squared_distances = True):
  
    if verbrose == True:
        print ("Alfa parameter: " + str(alfa))
        print ("Number of instances: " + str(len(V)))
        print ("Number of partitions: " + str(k)) 

    start_time2 = time.time()

    if initial_centroids == None:
        # Phase 1: Choosing initial centers with d^alfa-sampling
        C, _, _, _, _ = initialize(V, d, k, alfa, sort_by_distance = False)
    else:
        C = initial_centroids
  
    # Phase 2: Lloyd's algorithm
    t = 0
    while t < T_max:
        start_time = time.time()
        C_prim = []
        Voronoi_tiling, clus_i = get_Voronoi_tiling(C, V, k, d, beta)

        if verbrose == True:
            print ("Iteratie " + str(t))
            progress_bar.printProgressBar(0, k, "Progress:", "Complete", length = 50)
        for i in range(k):
#             instance_x = min(V, key = lambda x: sum(list(map(lambda v: np.linalg.norm(np.subtract(x, v)) ** beta, Voronoi_tiling[i]))))
            if sum_of_squared_distances == True:
                instance_x = min(V, key = lambda x: sum(list(map(lambda v: np.linalg.norm(np.subtract(x, v)) ** beta, Voronoi_tiling[i]))))
                C_prim.append(instance_x)
            else:
                C_prim.append(np.mean(Voronoi_tiling[i], axis = 0)) if len(Voronoi_tiling[i]) != 0 else C_prim.append(C[i])
            
            if verbrose == True:
                progress_bar.printProgressBar(i + 1, k, "Progress:", "Complete", length = 50)
        t += 1

        if verbrose == True:
            print ("Timp: " + str(time.time() - start_time))
        # print (clus_i)
        if all_equal(C_prim, C):
            break
        else:
            C = C_prim

    if verbrose == True:
        print ("Lloyds ended...")
        print ("Timp: (s)", time.time() - start_time2)
    # print (clus_i)
    return np.asarray(C), Voronoi_tiling, clus_i