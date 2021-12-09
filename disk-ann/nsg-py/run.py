import os
import re
import sys

import numpy as np

from random import randint

from tqdm import tqdm

from medioids import get_medioids
from nng import union_nng
from nsg import build_NSG,strongly_connect_NSG,query_NSG,save_NSG,load_NSG
from utils import fvecs_read,setup_data_dir,calculate_centroids,get_euclid_dist


def build_nsg_index():
    args = sys.argv
    PATH_TO_DATA = None

    if (len(args) != 2):
        raise Exception("Error: Supply relative path to data directory")
    else:
        PATH_TO_DATA = args[1]
    
    setup_data_dir(PATH_TO_DATA)
    
    path_to_shards = os.path.join(PATH_TO_DATA,"shards")

    union_nng(path_to_shards,4)

    centroids = calculate_centroids(path_to_shards)
    medioids = get_medioids(path_to_shards,centroids)


    fvecs = sorted(
                    [f for f in os.listdir(path_to_shards) if os.path.isfile(os.path.join(path_to_shards,f))],
                    key=lambda x: int(re.sub(r'[^\d]+','',x))
                )

    for i,shard_file in enumerate(fvecs):
        print("Constructing NSG Index for shard:",i+1)
        shard = fvecs_read(os.path.join(path_to_shards,shard_file))
        nng = np.load(os.path.join(path_to_shards,"../nng","nng_shard_"+str(i+1)+".npy"))
        nsg = build_NSG(shard,nng,int(medioids[i][1]),500,5)
        nsg = strongly_connect_NSG(nsg,shard,nng,int(medioids[i][1]),500,5)

        nsg_path = os.path.join(PATH_TO_DATA,"nsg")
        nsg_filename = "nsg_shard_"+str(i+1)+".nsg"

        save_NSG(nsg,nsg_path,nsg_filename)
    
    print("NSG Index Construction Complete!!")
    return

def query(q,k,path_to_shards):
    medioids = get_medioids(path_to_shards,None)
    centroids = calculate_centroids(path_to_shards)

    dists_to_centroids = [get_euclid_dist(q,centroid) for centroid in centroids]
    # print("Centroid Dists",dists_to_centroids)

    shard_idx = np.argmin(dists_to_centroids) + 1
    
    N = int(medioids[shard_idx-1][1])

    S = fvecs_read(os.path.join(path_to_shards,"sift_shard"+str(shard_idx)+".fvecs"))
    NSG = load_NSG(os.path.join(path_to_shards,"../nsg"),"nsg_shard_"+str(shard_idx)+".nsg")

    res = query_NSG(q,NSG,S,N)
    return res[:k]

def test(path_to_data):
    path_to_shards = os.path.join(path_to_data,"shards")
    all_data = fvecs_read(os.path.join(path_to_data,"siftsmall_base.fvecs"),record_dtype=np.float32)
    truth = fvecs_read(os.path.join(path_to_data,"siftsmall_groundtruth.ivecs"),record_dtype=np.float32)
    test_queries = fvecs_read(os.path.join(path_to_data,"siftsmall_query.fvecs"),record_dtype=np.float32)

    main_accs = []

    for i,q in enumerate(test_queries):
        vecs = query(q,5,path_to_shards)
        # print("Final Vecc",vecs)
        # break
        accs = []
        for (vec,test_vec_idx) in zip(vecs,truth[i][:5]):
            test_vec = all_data[int(test_vec_idx)]
            cor = 0
            for j in range(len(vec)):
                if (vec[j]==int(test_vec[j])):
                    cor+=1
            acc = cor/len(vec)
            accs.append(acc)
        main_accs.append(sum(accs)/5)
    
    return sum(main_accs)/len(main_accs)
            

def main(mode=1,q=None,k=None):
    if (mode==1):
        print("Running in Build Mode")
    elif(mode==2):
        print("Running in Query Mode")
    else:
        print("Running in Test Mode")

    args = sys.argv
    PATH_TO_DATA = None

    if (len(args) != 2):
        raise Exception("Error: Supply relative path to data directory")
    else:
        PATH_TO_DATA = args[1]
    
    if (mode==1):
        build_nsg_index()
    elif(mode==2):
        path_to_shards = os.path.join(PATH_TO_DATA,"shards")

        q_coll = fvecs_read(os.path.join(PATH_TO_DATA,"siftsmall_query.fvecs"))
        # q_coll = q_coll * 1e-40
        q = q_coll[randint(0,len(q_coll)-1)]

        print(query(q,k,path_to_shards))
    
    print(test(PATH_TO_DATA))

def shardTest():
    args = sys.argv
    PATH_TO_DATA = None

    if (len(args) != 2):
        raise Exception("Error: Supply relative path to data directory")
    else:
        PATH_TO_DATA = args[1]
    
    path_to_shards = os.path.join(PATH_TO_DATA,"shards")
    all_data = fvecs_read(os.path.join(PATH_TO_DATA,"siftsmall_base.fvecs"),record_dtype=np.float32)
    truth = fvecs_read(os.path.join(PATH_TO_DATA,"siftsmall_groundtruth.ivecs"),record_dtype=np.float32)
    test_queries = fvecs_read(os.path.join(PATH_TO_DATA,"siftsmall_query.fvecs"),record_dtype=np.float32)
    
    vec_to_shard = {}

    fvecs = sorted(
                    [f for f in os.listdir(path_to_shards) if os.path.isfile(os.path.join(path_to_shards,f))],
                    key=lambda x: int(re.sub(r'[^\d]+','',x))
                )

    for ts in tqdm(truth):
        for t in ts:
            if (t in vec_to_shard):
                continue
            vec = all_data[int(t)]
            for i,shard_file in enumerate(fvecs):
                found = False
                shard = fvecs_read(os.path.join(path_to_shards,shard_file))
                for s in shard:
                    dist = get_euclid_dist(vec,s)
                    if (dist==0):
                        vec_to_shard[t] = i+1
                        found = True
                        break
                if (found):
                    break
    
    return vec_to_shard

if __name__ == '__main__':
    # main(mode=1)
    # main(mode=3,k=5)
    # main(mode=3)
    print(shardTest())