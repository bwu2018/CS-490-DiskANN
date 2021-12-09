import numpy as np
import json
import glob
import re
import os
import pickle
import time


from copy import deepcopy

from utils import fvecs_read,calculate_centroids,get_euclid_dist
from nsg import load_NSG,query_NSG
from medioids import get_medioids



def search(query,path_to_shards,k=5,num_shards_query=2):

    # TODO: Can sort and store indexable filenames for easy access
    # Get closest centroids to query


    centroids = calculate_centroids(path_to_shards)
    distances = np.linalg.norm(centroids - query, axis = 1)
    distance_index = distances.argsort()[:num_shards_query]
    # print("Distance index:", distance_index)

    medoids = get_medioids(path_to_shards,centroids)
    medoid = medoids[distance_index[0]][1]

    # Get shards from candidate centroids
    all_shard_files = [f for f in os.listdir(path_to_shards) if os.path.isfile(os.path.join(path_to_shards,f))]
    shard_file_names = []
    for index in distance_index:
        for x in all_shard_files:
            if re.search("sift_shard" + str(index + 1) + "\.fvecs", x):
                shard_file_names.append(x)
    # print(shard_file_names)

    # Get shard graphs from candidate centroids
    path_to_nsg_idx = os.path.join(path_to_shards,"../nsg")
    all_shard_index = os.listdir(path_to_nsg_idx)
    shard_graph_names = []
    for index in distance_index:
        for x in all_shard_index:
            if re.search("nsg_shard_" + str(index + 1) + "\.nsg", x):
                shard_graph_names.append(x)
    # print(shard_graph_names)

    # TODO: Don't hard code merge of 2 shards
    P = fvecs_read(os.path.join(path_to_shards,shard_file_names[0]))
    G = load_NSG(os.path.join(path_to_shards,"../nsg"),shard_graph_names[0])
    # G = json.load(f)

    # f.close()

    P_merge = fvecs_read(os.path.join(path_to_shards,shard_file_names[1]))
    G_merge = load_NSG(os.path.join(path_to_shards,"../nsg"),shard_graph_names[1])
    # G_merge = json.load(f)
    # f.close()

    # Create 2nd shard index to merged index lookup
    reverse_lookup = dict()
    for i, x in enumerate(P):
        for j, y in enumerate(P_merge):
            if np.array_equal(x,y):
                G[i].add(-j)
                reverse_lookup[j] = i
                # print(G[i])
        # if i not in shard_to_merge.keys():
        #     np.append(P, x)
        #     shard_to_merge[i] = len(P)
        #     G[len(P)] = []

    

    # orig_keys = deepcopy(list(G.keys()))

    # for key in orig_keys:
    #     if (key in shard_to_merge):
    #         G[(1,key)] = set([])

    #     out_nodes = G_merge[key]
    #     true_out_nodes = [shard_to_merge[x] for x in out_nodes]
    #     # Cast to set -> list to remove duplicate edges
    #     G[shard_to_merge[key]] = list(set(list(G[shard_to_merge[key]]) + true_out_nodes))

    # print(G)

    # Search merged graph
    # result_size = 5
    # search_list_size = 60
    # print(P[154],P_merge[154])
    # print(P[224],P_merge[224])

    ann_res = query_NSG(query,G,P,int(medoid),P_merge,G_merge,reverse_lookup)
    # print(ann_res[:k])
    res = []
    for idx,dist in ann_res:
        if (idx < 0):
            res.append(P_merge[idx])
        else:
            res.append(P[idx])
    return res
    # print(greedy_search(P, G, start_node, query, result_size, search_list_size))

def eval(res):
    ground_truth = fvecs_read("../../siftsmall/siftsmall_groundtruth.ivecs")
    queries = fvecs_read("../../siftsmall/siftsmall_query.fvecs", record_dtype=np.float32)
    base = fvecs_read("../../siftsmall/siftsmall_base.fvecs", record_dtype=np.float32)
    metrics = []
    # with open("../../siftsmall/shards/medoids_index.pickle", "rb") as f:
    #     medoids_index = pickle.load(f)
    
    total_time = 0
    total_recall = 0
    for i in range(len(queries)):
        print(i)
        start_time = time.time()
        result = search(queries[i],"../../siftsmall/shards")
        time_spent = time.time() - start_time
        intersection = 0
        # Manually check to see if groundtruth in returned vectors
        for v in ground_truth[i]:
            for result_vector in result:
                if np.array_equiv(result_vector, base[v]):
                    intersection += 1
                    break
        total_time += time_spent
        total_recall += intersection / 100
        metrics.append([time_spent, intersection / 100])
    with open("metrics_5.pickle", "wb") as f:
        pickle.dump(metrics, f, pickle.HIGHEST_PROTOCOL)
    print("Average time:", total_time / 100)
    print("Average recall:", total_recall / 100)