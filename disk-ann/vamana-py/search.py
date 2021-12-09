import numpy as np
import pickle
import glob
import re
from utils import fvecs_read, greedy_search


def main():
    num_shards_query = 2

    query = []
    
    # TODO: Can sort and store indexable filenames for easy access
    # Get closest centroids to query
    centroids = np.loadtxt("../sift/shards/centroids.txt", delimiter = ",")
    distances = np.linalg.norm(centroids - query, axis = 1)
    distance_index = distances.argsort()[:num_shards_query]
    print("Distance index:", distance_index)

    # Get shards from candidate centroids
    all_shard_files = glob.glob("../sift/shards/*.fvecs")
    shard_file_names = []
    for index in distance_index:
        for x in all_shard_files:
            if re.search("shard" + str(index + 1) + "\.fvecs", x):
                shard_file_names.append(x)
    print(shard_file_names)

    # Get shard graphs from candidate centroids
    all_shard_index = glob.glob("../sift/shards/vamana_indexes/*.pickle")
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

    P_merge = fvecs_read(shard_file_names[1])
    with open(shard_graph_names[1], 'rb') as f:
        G_merge = pickle.load(f)

    # Create lookup dictionary mapping 2nd shard index to the merged shard index
    shard_to_merge = dict()
    for i, x in enumerate(P_merge):
        for j, y in enumerate(P):
            if x == y:
                shard_to_merge[i] = j
        if i not in shard_to_merge.keys():
            P.append(x)
            shard_to_merge[i] = len(P)

    # Merge shard graphs using merged index
    for key in G_merge.keys():
        out_nodes = G_merge[key]
        true_out_nodes = [shard_to_merge[x] for x in out_nodes]
        # Cast to set -> list to remove duplicate edges
        G[shard_to_merge[key]] = list(set(G[shard_to_merge[key]] + true_out_nodes))

    # Search merged graph
    result_size = 5
    search_list_size = 60
    print(greedy_search(P, G, start_node, query, result_size, search_list_size))


if __name__ == '__main__':
    main()