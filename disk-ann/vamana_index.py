import numpy as np
from collections import defaultdict
import random


def n_closest_points(P: np.array, query: np.array, query_set: set, n: int) -> list:
    query_list = list(query_set)
    raw_query_list = np.asarray([P[x] for x in query_list])
    distances = np.linalg.norm(raw_query_list - query, axis=1)
    distance_index = distances.argsort()[:n]
    return [query_list[x] for x in distance_index]

# Graph G
def greedy_search(P: np.array, G: defaultdict, start_node: int, query: np.array, result_size: int, search_list_size: int) -> list:
    result = set()
    result.add(start_node)
    visited = set()
    while len(result.difference(visited)) != 0:
        diff = result.difference(visited)
        temp_p = list(diff)[0]
        for point in diff:
            if np.linalg.norm(P[point] - query) < np.linalg.norm(P[temp_p] - query):
                temp_p = point
        result = result.union(set(G[temp_p]))
        visited.add(temp_p)
        if len(result) > search_list_size:
            result = set(n_closest_points(P, query, result, search_list_size))
    return n_closest_points(P, query, result, result_size)

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

def vamana(shard_filename, alpha, L, R):
    # P is list of points in shard
    P = [
        [1,1],
        [2,2],
        [100,100],
        [200,200],
        [300,300],
        [400,400],
        [500,500]
    ]
    P = np.asarray([np.asarray(x) for x in P])
    print(P)

    G = defaultdict(list)
    index_list = [x for x in range(len(P))]
    for i in index_list:
        G[i] = random.sample(index_list[:i] + index_list[i+1:], R)

    print("Random Graph:", G)

    # TODO: Find actual mediod
    mediod = len(P) // 2 # Index of mediod
    print("Mediod:", mediod)
    sigma = [x for x in range(len(P))]
    random.shuffle(sigma)

    # print([P[x] for x in greedy_search(P, G, mediod, np.asarray([500,400]), 3, L)])
    # exit()

    for i in sigma:
        print("i:", i)
        V = set(greedy_search(P, G, mediod, P[i], 1, L))
        print(V)
        print(G)
        G = robust_prune(P, G, i, V, alpha, R)
        print(G)
        for point in G[i]:
            temp_set = set(G[point])
            temp_set.add(i)
            if len(temp_set) > R:
                G = robust_prune(P, G, point, temp_set, alpha, R)
            else:
                G[point] = list(temp_set)
    
    # Save graph G
    print(G)

def main():
    random.seed(10)
    vamana("temp", 1, 8, 2)


if __name__ == '__main__':
    main()
