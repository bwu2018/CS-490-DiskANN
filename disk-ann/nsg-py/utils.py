import numpy as np
import os
import re

from random import randint

from sklearn.cluster import KMeans


N_SHARDS = 40
CHUNK_SIZE = 1000
N_DIM = 128
DATA_SIZE = 4
N_CLOSEST_CENTERS = 2

PATH_TO_DATA = '../sift/'


def my_mkdr(path):
    if (os.path.exists(path)):
        print("The Path:",path,"already exists!")
        return
    os.mkdir(path)

def setup_data_dir(path):
    my_mkdr(path)
    my_mkdr(os.path.join(path,"shards"))
    my_mkdr(os.path.join(path,"nng"))
    my_mkdr(os.path.join(path,"centroids"))
    my_mkdr(os.path.join(path,"nsg"))

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

def calculate_centroids(path_to_shards):
    if (os.path.exists(os.path.join(path_to_shards,"../centroids/centroids.npy"))):
        return np.load(os.path.join(path_to_shards,"../centroids/centroids.npy"))
    fvecs = sorted(
                    [f for f in os.listdir(path_to_shards) if os.path.isfile(os.path.join(path_to_shards,f))],
                    key=lambda x: int(re.sub(r'[^\d]+','',x))
                )
    
    centroids = []

    for i,shard_file in enumerate(fvecs):
        print("Calculating centroid for shard",i+1)
        shard = fvecs_read(os.path.join(path_to_shards,shard_file))
        kmeans = KMeans(n_clusters=1).fit(shard)
        centroids.append(kmeans.cluster_centers_[0])
    
    np.save(os.path.join(path_to_shards,"../centroids/centroids.npy"),centroids)
    
    return centroids

def get_euclid_dist(m,n):
    # return np.sqrt(sum([(i-j)**2 for i,j in zip(m,n)]))
    return np.linalg.norm(m-n)

def get_midpt(m,n):
    return (m+n)/2


def get_nn(q,S,G,navigating_node=None,traversal_limit=None,break_at_zero=True,return_visited_nodes=False):
    """
    Traverses Nearest Neighbors using Greedy BFS traversal.
    Required Arguments:
    q: query
    S: shard
    G: nearest neighbors graph (NNG)

    Optional:

    navigating_node: Node to start traversal, if None, start at random node. Default: None
    traversal_limit: Max number of nodes to search before ending traversal. If None, searches at most sqrt(len(S))*2 nodes. If "exact", searches entire graph. Default=None.
    break_at_zero: Boolean for indicating if traversal ends when it finds an exact match as early stopping measure. Default=True.
    return_visited_nodes: returns visited nodes if selected. Default=False
    """
    if (traversal_limit==None):
        traversal_limit = np.sqrt(len(S)) * 2
        
    if (traversal_limit=="exact"):
        traversal_limit = len(S)

    visited = set()
    visited_idx = set()
    
    current_idx = randint(0,len(S)-1) if not navigating_node else navigating_node
    bfs_queue = [(S[current_idx],current_idx)]
    cur_min = 1000000
    cur_min_idx = -1
    itr = 0
    while (len(visited) < traversal_limit and bfs_queue):
        itr += 1
        v,current_idx = bfs_queue.pop(0)
                
        adj = G[current_idx]
        
        dist = get_euclid_dist(q,v)
        visited.add((current_idx,dist))
        visited_idx.add(current_idx)
        
        cur_min_idx = cur_min_idx if (np.argmin([cur_min,dist]) == 0) else current_idx
        cur_min = min([cur_min,dist])
        
        if (dist > cur_min):
            continue
        if (cur_min==0.0 and break_at_zero):
            break
        
        for n in adj:
            if (n not in visited_idx):
                bfs_queue.append((S[n],n))

    if (return_visited_nodes):
        return cur_min,cur_min_idx,visited
    return cur_min,cur_min_idx

def create_open_ball(m,r):
    """
    Creates a dictionary to represent an openball with m as its center and r as its radius

    Required Arguments:
    m: vector to be the Open Ball's center
    r: radius of Open Ball
    """
    open_ball = {}
    open_ball["r"] = r
    open_ball["c"] = m
    return open_ball

def in_open_ball(m,open_ball):
    """
    Checks if m is in the open ball

    Required Arguments:
    m: vector
    open_ball: open ball from create_open_ball()
    """
    dist = get_euclid_dist(m,open_ball["c"])
    return dist < open_ball["r"]

def in_lune(m,open_ball_1,open_ball_2):
    """
    Checks if the vector m is in the intersection of the two open balls.

    Required Arguments:
    m: vector
    open_ball_1: first open ball from create_open_ball()
    open_ball_2: second open ball from create_open_ball()
    """
    return in_open_ball(m,open_ball_1) and in_open_ball(m,open_ball_2)