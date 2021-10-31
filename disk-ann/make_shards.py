import numpy as np
from sklearn.cluster import KMeans
import os
import sys
from collections import defaultdict

N_CLUSTERS = 3
CHUNK_SIZE = 1000
N_DIM = 128
DATA_SIZE = 4
N_CLOSEST_CENTERS = 2

PATH_TO_DATA = '../sift/'

def fvecs_read(filename, c_contiguous=True, record_count=-1, line_offset=0):
    fv = np.fromfile(filename, dtype=np.float32, count=record_count * (N_DIM + 1), offset=line_offset * (N_DIM + 1) * DATA_SIZE)
    if fv.size == 0:
        return np.zeros((0, 0))
    #print(fv)
    #print(fv.shape)
    dim = fv.view(np.int32)[0]
    assert dim > 0
    fv = fv.reshape(-1, 1 + dim)
    if not all(fv.view(np.int32)[:, 0] == dim):
        raise IOError("Non-uniform vector sizes in " + filename)
    fv = fv[:, 1:]
    if c_contiguous:
        fv = fv.copy()
    return fv

#train_data  = fvecs_read('../sift/sift_learn.fvecs')

#print(fvecs_read(PATH_TO_DATA + 'shards/sift_shard1.fvecs'))

train_data = fvecs_read(PATH_TO_DATA + 'sift_base.fvecs', record_count=CHUNK_SIZE)
num_base_vecs = int(os.stat(PATH_TO_DATA + 'sift_base.fvecs').st_size / (N_DIM + 1) / DATA_SIZE)

print(num_base_vecs)

kmeans = KMeans(n_clusters=N_CLUSTERS).fit(train_data)

centroids = kmeans.cluster_centers_

print(centroids)

centroid_lookup = defaultdict(list)

for i in range(num_base_vecs // CHUNK_SIZE):
    vectors = fvecs_read(PATH_TO_DATA + 'sift_base.fvecs', record_count=CHUNK_SIZE, line_offset=CHUNK_SIZE * i)
    for vec in vectors:
        distances = np.linalg.norm(centroids - vec, axis=1)
        closest_centers = distances.argsort()[:N_CLOSEST_CENTERS]
        for c in closest_centers:
            vec = np.insert(vec, 0, 1.8e-43, axis=0)
            centroid_lookup[c].append(vec)

for c in centroid_lookup:
    vector_shard = np.array(centroid_lookup[c])
    vector_shard = vector_shard.flatten()
    print(vector_shard.shape)
    vector_shard.tofile(PATH_TO_DATA + 'shards/sift_shard' + str(c + 1) + '.fvecs')

