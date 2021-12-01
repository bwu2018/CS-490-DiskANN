import os
import re

import numpy as np

from utils import fvecs_read,get_nn

def get_medioids(path_to_shards,centroids,check_for_existing=True):
    """
    Returns the Medioids for all shards using our NNG.

    Required Arguments:
    path_to_shards: path to shards folder
    centroids: list of centroids for each shard

    Optional Arguments:
    check_for_existing: If medioids exist as saved .npy files, return those. Default=True
    """
    if (check_for_existing and os.path.exists(os.path.join(path_to_shards,"../centroids/medioids.npy"))):
        return np.load(os.path.join(path_to_shards,"../centroids/medioids.npy"))
    fvecs = sorted(
                    [f for f in os.listdir(path_to_shards) if os.path.isfile(os.path.join(path_to_shards,f))],
                    key=lambda x: int(re.sub(r'[^\d]+','',x))
                )
    

    medioids = []
        
    for i,shard_file in enumerate(fvecs):
        print("Calculating medioid for shard",i+1)
        centroid = centroids[i]
        nng = np.load(os.path.join(path_to_shards,"../nng","nng_shard_"+str(i+1)+".npy"))
        shard = fvecs_read(os.path.join(path_to_shards,shard_file))
        cur_dist,cur_idx = get_nn(centroid,shard,nng)
        medioids.append((i+1,cur_idx,cur_dist))
    
    np.save(os.path.join(path_to_shards,"../centroids/medioids.npy"),np.array(medioids))
    
    return medioids