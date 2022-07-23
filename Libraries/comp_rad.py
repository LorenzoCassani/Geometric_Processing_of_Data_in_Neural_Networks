# Standard modules imports
import math

# Third-party modules imports
import torch.linalg as linalg


# Note 1: the arguments of every function defined below are either a torch.tensor list
# or a couple of torch.tensor lists;
# every torch.tensor list represents a point cloud / manifold.

# Note 2: N is the number of points / vectors that constitue the point cloud / manifold


# Manifold mean vector function
def mean_vector(data):
    n = len(data)
    
    # Normalization
    normalized_data = []
    for x in data:
        if not all(x == 0): # Non all-zero vector
            x_norm = linalg.vector_norm(x)
            x = x / x_norm
        normalized_data += [x]
    
    summation = sum(normalized_data)
    mean = summation / n
    return mean


# Manifolds centre-centre distance function
def cc_distance(data_1, data_2):
    distance = linalg.vector_norm(mean_vector(data_1) - mean_vector(data_2))
    return distance


# Manifold (normalized) radius of gyration function
def gyradius(data):
    n = len(data)
    gyradius = 0.
    
    # Normalization
    normalized_data = []
    for x in data:
        if not all(x == 0): # Non all-zero vector
            x_norm = linalg.vector_norm(x)
            x = x / x_norm
        normalized_data += [x]
    
    x_m = mean_vector(normalized_data)
    for x in normalized_data:
        gyradius += (linalg.vector_norm(x - x_m))**2
    gyradius = math.sqrt(gyradius / n)
    return gyradius