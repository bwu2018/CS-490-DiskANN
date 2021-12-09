import numpy as np
import random
import re
import pickle
import glob
import time
from utils import fvecs_read, greedy_search


def n_closest_points(P: np.array, query: np.array, query_set: set, n: int) -> list:
    query_list = list(query_set)
    raw_query_list = np.asarray([P[x] for x in query_list])
    distances = np.linalg.norm(raw_query_list - query, axis=1)
    distance_index = distances.argsort()[:n]
    return [query_list[x] for x in distance_index]

# Graph G, Point p, candidate set V, distance threshold alpha, degree bound R
# G is adjacency list
def robust_prune(P, G, p, V, alpha, R):
    for x in G[p]:
        V.add(x)
    V = V.difference({p})
    G[p] = []
    while len(V) != 0:
        list_V = list(V)
        temp_p = list_V[0]
        for point in list_V:
            if np.linalg.norm(P[point] - P[p]) < np.linalg.norm(P[temp_p] - P[p]):
                temp_p = point
        G[p].append(temp_p)
        if len(G[p]) == R:
            break
        to_remove = []
        for point in V:
            if alpha * np.linalg.norm(P[temp_p] - P[point]) <= np.linalg.norm(P[p] - P[point]):
                to_remove.append(point)
        for point in to_remove:
            V.remove(point)
    return G

def vamana_helper(P, G, alpha, L, R, medoid):
    # P is list of points in shard
    # P = [
    #     [1,1],
    #     [2,2],
    #     [100,100],
    #     [200,200],
    #     [300,300],
    #     [400,400],
    #     [500,500]
    # ]
    # P = np.asarray([np.asarray(x) for x in P])
    # print(P)

    sigma = [x for x in range(len(P))]
    random.shuffle(sigma)

    # print([P[x] for x in greedy_search(P, G, medoid, np.asarray([500,400]), 3, L)])
    # exit()

    for i in sigma:
        # print("i:", i)
        V = set(greedy_search(P, G, medoid, P[i], 1, L))
        # print(V)
        # print(G)
        G = robust_prune(P, G, i, V, alpha, R)
        # print(G)
        for point in G[i]:
            temp_set = set(G[point])
            temp_set.add(i)
            if len(temp_set) > R:
                G = robust_prune(P, G, point, temp_set, alpha, R)
            else:
                G[point] = list(temp_set)
    return G

def vamana(shard_filename, alpha, L, R):
    P = fvecs_read(shard_filename)

    # Create random R-regular graph
    G = dict()
    index_list = [x for x in range(len(P))]
    for i in index_list:
        G[i] = random.sample(index_list[:i] + index_list[i+1:], R)

    # print("Random Graph:", G)
    
    file_num = int(re.findall(r'\d+', shard_filename)[0])

    with open("../sift/shards/medoids_index.pickle") as f:
        medoids_index = pickle.load(f)
    medoid = medoids_index[str(file_num - 1)]

    G = vamana_helper(P, G, 1, L, R, medoid)
    G = vamana_helper(P, G, alpha, L, R, medoid)

    # Save graph G
    with open(f"../sift/shards/vamana_indexes/vamana_index{file_num}.pickle", "wb") as f:
        pickle.dump(G, f, protocol=pickle.HIGHEST_PROTOCOL)
    f.write(to_write)
    f.close()

def main():
    random.seed(10)
    alpha = 2
    L = 75
    R = 60

    files = glob.glob("../sift/shards/*.fvecs")
    start_time = time.time()
    for shard_filename in files:
        print(shard_filename)
        shard_start = time.time()
        vamana(shard_filename, alpha, L, R)
        print("Shard index time: " + str(time.time() - shard_start))
    print("Total Time Taken: " + str(time.time() - start_time))

if __name__ == '__main__':
    main()
