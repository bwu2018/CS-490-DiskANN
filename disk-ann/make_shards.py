import numpy as np
from sklearn.cluster import KMeans
import os
import sys
import glob
from collections import defaultdict
import time

N_SHARDS = 3
CHUNK_SIZE = 1000
N_DIM = 128
DATA_SIZE = 4
N_CLOSEST_CENTERS = 2

PATH_TO_DATA = '../sift/'

args = [int(arg) for arg in sys.argv[1:]]
if len(args) == 1:
    N_SHARDS = args[0]
elif len(args) == 2:
    N_SHARDS, N_CLOSEST_CENTERS = args
elif len(args) == 4:
    N_SHARDS, N_CLOSEST_CENTERS, N_DIM, CHUNK_SIZE = args


def fvecs_read(filename, c_contiguous=True, record_count=-1, line_offset=0, record_dtype=np.float32):
    if record_count > 0:
        record_count *= N_DIM + 1
    if line_offset > 0:
        line_offset *= (N_DIM + 1) * DATA_SIZE
    fv = np.fromfile(filename, dtype=record_dtype, count=record_count, offset=line_offset)
    if fv.size == 0:
        return np.zeros((0, 0))
    dim = fv.view(np.int32)[0]
    #print(dim)
    assert dim > 0
    fv = fv.reshape(-1, 1 + dim)
    if not all(fv.view(np.int32)[:, 0] == dim):
        raise IOError("Non-uniform vector sizes in " + filename)
    fv = fv[:, 1:]
    if c_contiguous:
        fv = fv.copy()
    return fv

def main():
    start_time = time.time()
    
    train_data = fvecs_read(PATH_TO_DATA + 'sift_base.fvecs', record_count=CHUNK_SIZE)
    num_base_vecs = int(os.stat(PATH_TO_DATA + 'sift_base.fvecs').st_size / (N_DIM + 1) / DATA_SIZE)

    print('{} base vectors'.format(num_base_vecs))

    kmeans = KMeans(n_clusters=N_SHARDS).fit(train_data)

    centroids = kmeans.cluster_centers_

    print('KMeans Centroids:')
    print(centroids)

    centroid_lookup = defaultdict(list)

    for i in range(num_base_vecs // CHUNK_SIZE):
        vectors = fvecs_read(PATH_TO_DATA + 'sift_base.fvecs', record_count=CHUNK_SIZE, line_offset=CHUNK_SIZE * i)
        for vec in vectors:
            distances = np.linalg.norm(centroids - vec, axis=1)
            closest_centers = distances.argsort()[:N_CLOSEST_CENTERS]
            for c in closest_centers:
                #vec = np.insert(vec, 0, 1.8e-43, axis=0)
                centroid_lookup[c].append(vec)

    # delete existing shards
    
    files = glob.glob(PATH_TO_DATA + 'shards/*.fvecs')
    for f in files:
        os.remove(f)    

    for i, c in enumerate(centroid_lookup):
        print('Writing shard ' + str(i))
        shard_start = time.time()
        vector_shard = np.array(centroid_lookup[c])
        vector_shard = vector_shard.astype(np.int32)
        zero_col = np.full((len(vector_shard), 1), N_DIM, dtype=np.int32)
        vector_shard = np.c_[zero_col, vector_shard]
        print('Shard shape: {}'.format(vector_shard.shape))
        vector_shard.tofile(PATH_TO_DATA + 'shards/sift_shard' + str(c + 1) + '.fvecs')
        print('Shard write time:', time.time() - shard_start)
    
    print('Total Time Taken:', time.time() - start_time)

if __name__ == '__main__':
    main()
