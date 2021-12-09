import numpy as np


def fvecs_read(filename, c_contiguous=True, record_count=-1, line_offset=0, record_dtype=np.int32):
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

def greedy_search(P: np.array, G: dict, start_node: int, query: np.array, result_size: int, search_list_size: int) -> list:
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