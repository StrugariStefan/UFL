from functools import reduce
import numpy as np

def get_distances(C, V, d):
    distances = []
    for instance in V:
        min_distance = reduce(lambda x,y: min(x, y), [np.linalg.norm(np.subtract(instance, centroid), d) if any(centroid != instance) else 0.0 for centroid in C], np.inf)
        distances.append(min_distance)
    return distances

def d_alfa_sample(C, V, d, alfa, z, sort_by_distance = False, distances = []):
    distances = []
    intervals = [0.0]
    
    if len(C) == 0:
        intervals = [(i + 1)/(len(V)) for i in range(len(V))]
    else:
        if distances == []:
            distances = get_distances(C, V, d)
        d_total = sum(distances)
        if sort_by_distance == True:
            distances_i = [dist[0] for dist in sorted(enumerate(distances), key = lambda element: element[1],  reverse = True)]
            V = V[distances_i]
            distances.sort(reverse = True)
        distances = [dist ** alfa if dist != 0 else 0.0 for dist in distances]
        cumdistances = np.cumsum(distances)
        intervals = [cumdist / cumdistances[-1]  for cumdist in cumdistances]

    c = reduce(lambda value, element: V[element[0]+1] if z >= element[1] else value, list(enumerate(intervals[:-1])), V[0])
    C.append(c)
    for i, v in enumerate(V):
        if all(v == c):
            instance_index = i
            break
    return [0.0] + intervals, V, C, instance_index