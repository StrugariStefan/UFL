import queue
from d_sampling import get_distances, d_alfa_sample
from math import log10
from math import floor

def binary_search(C, A, V, d, epsilon, instance_index, zt, distances, last = None):
    alfa = (A[1] + A[0]) / 2
    intervals, V, C_prim, _ = d_alfa_sample(C[:], V, d, alfa, False, distances = distances)
    if abs(intervals[instance_index] - zt) < epsilon or (abs(last - intervals[instance_index]) < epsilon if last != None else False):
        return round(alfa, floor(abs(log10(epsilon))))
    elif intervals[instance_index] < zt:
        return binary_search(C, (alfa, A[1]), V, d, epsilon, instance_index, zt, distances, intervals[instance_index])
    else:
        return binary_search(C, (A[0], alfa), V, d, epsilon, instance_index, zt, distances, intervals[instance_index])

def algorithm2(V, d, k, Z, alfa_h, epsilon, explicit = False):
    """
    Yields all posible pairs (C, A) for the given instance (V, Z), where C is
    the set of initial centroids for any alpha parameter in interval A. This
    method is meant to reduce the alpha space needed to test in order to
    determine the best alpha parameter for the given dataset.

    @params:
        V               - Required  : clustering sample ()
        d               - Required  : distance metric (Int)
        k               - Required  : desiered number of clusters (Int)
        Z               - Required  : uniform random vector, of size k ()
        alfa_h          - Required  : superior end of the searching interval (Int)
        epsilon         - Required  : breakpoint determing precision (Float)
        explicit        - Optional  : also yield  (Boolean)
    @yield:
        (C, A)          - initial centroids and the coresponding alpha interval
        (:obj:`list`, :obj:`tuple`)            
    """


    Q = queue.Queue()
    Q.put(([], (0, alfa_h)))

    while Q.empty() == False:
        C, A = Q.get()
        t = len(C)
        z = Z[t]
        
        V_prim = V[:]

        intervals_min, V_prim, _, instance_index_inf = d_alfa_sample(C[:], V_prim, d, A[0], z, True)
        distances = get_distances(C, V_prim, d)
        intervals_max, V_prim, _, instance_index_sup = d_alfa_sample(C[:], V_prim, d, A[1], z, distances = distances)
        
        a_inf = A[0]
        for i in range(instance_index_inf, instance_index_sup, -1):
            alfa = binary_search(C, A, V_prim, d, epsilon, i, z, distances)
            A_i = (a_inf, alfa)
            a_inf = alfa 
            C_i = C[:]
            C_i.append(V_prim[i])
            
            if len(C_i) < k:
                Q.put((C_i, A_i))
                if explicit:
                    yield (C_i, A_i)
            else:
                yield (C_i, A_i)
        else:        
            A_i = (a_inf, A[1])
            C_i = C[:]
            C_i.append(V_prim[instance_index_sup])

            if len(C_i) < k:
                Q.put((C_i, A_i))
                if explicit:
                    yield (C_i, A_i)
            else:
                yield (C_i, A_i)