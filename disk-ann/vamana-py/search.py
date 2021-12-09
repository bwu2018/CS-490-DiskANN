import numpy as np
import pickle
import glob
import re
from utils import fvecs_read, greedy_search
import time


def search(query, medoids_index, num_shards_query=2):
    
    # TODO: Can sort and store indexable filenames for easy access
    # Get closest centroids to query
    centroids = np.loadtxt("../../siftsmall/shards/centroids.txt", delimiter = ",")
    distances = np.linalg.norm(centroids - query, axis = 1)
    distance_index = distances.argsort()[:num_shards_query]
    print("Distance index:", distance_index)

    # Get shards from candidate centroids
    all_shard_files = glob.glob("../../siftsmall/shards/*.fvecs")
    shard_file_names = []
    for index in distance_index:
        for x in all_shard_files:
            if re.search("shard" + str(index + 1) + "\.fvecs", x):
                shard_file_names.append(x)
    print(shard_file_names)

    # Get shard graphs from candidate centroids
    all_shard_index = glob.glob("../../siftsmall/shards/vamana_indexes/*.pickle")
    shard_graph_names = []
    for index in distance_index:
        for x in all_shard_index:
            if re.search("index" + str(index + 1) + "\.pickle", x):
                shard_graph_names.append(x)
    print(shard_graph_names)

    # TODO: Don't hard code merge of 2 shards
    P = fvecs_read(shard_file_names[0])
    with open(shard_graph_names[0], 'rb') as f:
        G = pickle.load(f)
    assert(len(P) == len(G.keys()))

    P_merge = fvecs_read(shard_file_names[1])
    with open(shard_graph_names[1], 'rb') as f:
        G_merge = pickle.load(f)

    # Create lookup dictionary mapping 2nd shard index to the merged shard index
    shard_to_merge = dict()
    for i, x in enumerate(P_merge):
        for j, y in enumerate(P):
            if np.array_equiv(x, y):
                shard_to_merge[i] = j
        if i not in shard_to_merge.keys():
            P = np.vstack([P, x])
            shard_to_merge[i] = len(P) - 1
            G[len(P) - 1] = []

    # Merge shard graphs using merged index
    for key in G_merge.keys():
        out_nodes = G_merge[key]
        true_out_nodes = [shard_to_merge[x] for x in out_nodes]
        # Cast to set -> list to remove duplicate edges
        G[shard_to_merge[key]] = list(set(G[shard_to_merge[key]] + true_out_nodes))
    assert(len(P) == len(G))

    # Search merged graph
    result_size = 100
    search_list_size = 100
    index_results = greedy_search(P, G, medoids_index[distance_index[0]], query, result_size, search_list_size)
    return [P[x] for x in index_results]

def main():
    ground_truth = fvecs_read("../../siftsmall/siftsmall_groundtruth.ivecs")
    queries = fvecs_read("../../siftsmall/siftsmall_query.fvecs", record_dtype=np.float32)
    base = fvecs_read("../../siftsmall/siftsmall_base.fvecs", record_dtype=np.float32)
    metrics = []
    with open("../../siftsmall/shards/medoids_index.pickle", "rb") as f:
        medoids_index = pickle.load(f)
    
    total_time = 0
    total_recall = 0
    for i in range(len(queries)):
        print(i)
        start_time = time.time()
        result = search(queries[i], medoids_index)
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

if __name__ == '__main__':
    main()