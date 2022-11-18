from cupy_knn import LBVHIndex
from cupyx.profiler import benchmark
import numpy as np

lbvh = LBVHIndex()

points = np.random.randn(1000, 3)

# build the index
lbvh.build(points)

# prepare the index for knn search with K=16
lbvh.prepare_knn_default(16)

query_points = np.random.randn(1000, 3)
# do one query for each of the points in the dataset
lbvh.query_knn(query_points)