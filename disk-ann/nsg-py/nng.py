import numpy as np
import sys
import os
import glob
import time
import subprocess
import re
import pickle

from sklearn.neighbors import kneighbors_graph

from utils import fvecs_read

N_SHARDS = 40
CHUNK_SIZE = 1000
N_DIM = 128
DATA_SIZE = 4
N_CLOSEST_CENTERS = 2

PATH_TO_DATA = '../../sift/'

def build_vector_idx(path_to_shards):
    fvecs = sorted(
                    [f for f in os.listdir(path_to_shards) if os.path.isfile(os.path.join(path_to_shards,f))],
                    key=lambda x: int(re.sub(r'[^\d]+','',x))
                )

  
    idx = []

    for i,shard_file in enumerate(fvecs):
        shard = fvecs_read(os.path.join(path_to_shards,shard_file))
        for j,vec in enumerate(shard):
            vec_idx = hex(i)+"-"+hex(j)
            idx.append(vec_idx)
    
    return idx


def build_nng_shard(shard,k=1):
    nng_sk = kneighbors_graph(shard,n_neighbors=k,n_jobs=2)
    nng = np.array([np.flatnonzero(n.toarray() == np.max(n.toarray())) for n in nng_sk])
    return nng

def union_nng(path_to_shards,k):
    fvecs = sorted(
                    [f for f in os.listdir(path_to_shards) if os.path.isfile(os.path.join(path_to_shards,f))],
                    key=lambda x: int(re.sub(r'[^\d]+','',x))
                )

  
    nngs = []

    for i,shard_file in enumerate(fvecs):
        if (i+1 <= 22):
            continue
        print("Building nng for Shard:",i+1)
        shard = fvecs_read(os.path.join(path_to_shards,shard_file))
        nng = build_nng_shard(shard,k)
        save_nng(os.path.join(path_to_shards,"../nng/","nng_shard_"+str(i+1)),nng)
#     return nngs

def save_nng(filename,nng):
    print("Saving NNG to:",filename)
    np.save(filename,nng)


# nng = union_nng(PATH_TO_DATA+"shards")
# save_nng(nng,PATH_TO_DATA+"nng/nng.npy")