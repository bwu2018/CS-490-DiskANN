import numpy as np

import pickle
import os

from random import randint

from tqdm import tqdm

from utils import get_euclid_dist,get_nn,create_open_ball,in_lune

def check_edge(m,points_in_lune,G):
    """
    Returns True if edge should connect from p to m using MRNG edge selection strategy.

    Arguments:
    m: node to point to.
    points_in_lune: collection of nodes within the lune(pm)
    G: graph in adjacency list format.
    """
    if (not points_in_lune):
        return True
    if (G):
        for v in points_in_lune:
            if (v in G[m]):
                return False
    return True
    
def edge_selection(p,p_idx,S,g={},m=None):
    """
    Performs MRNG edge selection strategy. 
    Returns graph after pruning.

    For more details on edge selection refer to Definition 5 here: https://www.vldb.org/pvldb/vol12/p461-fu.pdf#page=6

    Required Arguments:
    p: Node to create outgoing edges for
    p_idx: Index corresponding to vector p (refer to S)
    S: Array of nodes to check for edge selection. In format [(vector,idx),...]
    
    Optional Arguments:\n
    g: Graph in adjacency list format. Default: Empty graph
    m: outdegree limit. If None outdegree limit is len(S)-1 (highly suggest against this). Default=None.
    """
    if (not m):
        m=len(S)-1
    mjr_ct = 0
    edges = 0
    for q,i in S:
        if (edges == m):
            break
        if (i==p_idx):
            continue
        mjr_ct += 1
        dist = get_euclid_dist(p,q)
        open_ball_p = create_open_ball(p,dist)
        open_ball_q = create_open_ball(q,dist)
        
        points_in_lune = []
        mnr_ct = 0
        for r,j in S:
            mnr_ct += 1
            if (mnr_ct >= mjr_ct):
                break
            if (in_lune(r,open_ball_p,open_ball_q)):
                if (i==j or i==p_idx):
                    continue
                points_in_lune.append(j)
        if (check_edge(p_idx,points_in_lune,g)):
            edges+=1
            if (p_idx not in g):
                g[p_idx] = set([i])
            else:
                g[p_idx].add(i) 
    return g

def build_NSG(S,G,navigating_node,l,m,use_tqdm=False):
    """
    Builds main parts of NSG.

    Required Arguments:
    S: Shard
    G: NNG
    navigating_node: The navigating node vector. This is the medioid of the shard.
    l: Traversal Limit for NNG
    m: Max outdegree of nodes.

    Optional Arguments:
    use_tqdm: Use progressbar to see progress. Default=False.
    """


    NSG = {}

    itr = enumerate(S)

    if (use_tqdm):
        itr = tqdm(itr)

    for i,v in itr:
        _,_,E = get_nn(v,S,G,navigating_node,traversal_limit=l,return_visited_nodes=True)
        E = sorted(E,key=lambda x: x[1])
        E = [(S[idx],idx) for idx,_ in E]
        
        NSG = edge_selection(v,i,E,g=NSG,m=m)
    
    return NSG

def strongly_connect_NSG(NSG,S,G,navigating_node,l,m,use_tqdm=False):
    """
    Strongly connects NSG by connecting navigating node, and nodes without indegrees to it's nearest neighbor.

    Required Arguments:
    NSG: NSG built from build_NSG()
    S: Shard
    G: NNG
    navigating_node: The navigating node vector. This is the medioid of the shard.
    l: Traversal Limit for NNG
    m: Max outdegree of nodes.

    Optional Arguments:
    use_tqdm: Use progressbar to see progress. Default=False.
    """
   
    _,_,E = get_nn(S[navigating_node],S,G,navigating_node,traversal_limit=l,break_at_zero=False,return_visited_nodes=True)
    E = sorted(E,key=lambda x: x[1])
    E.pop(0)
    
    ct = 0
    for i,_ in E:
        ct += 1
        if (ct == m):
            break
        if (navigating_node not in NSG):
            NSG[navigating_node] = {i}
        else:
            NSG[navigating_node].add(i)
    
    indegree_nodes = {}
    for k,v in NSG.items():
        v.union(v)

    itr = NSG.keys()

    if (use_tqdm):
        itr = tqdm(itr)

    for v in itr:
        if (v not in indegree_nodes):
            _,_,E = get_nn(S[v],S,G,navigating_node,traversal_limit=l,return_visited_nodes=True)
            E = sorted(E,key=lambda x: x[1])
            E = [idx for idx,_ in E]
            
            while(E):
                p = E.pop(0)
                if (p==v):
                    continue
                if (p not in NSG):
                    NSG[p] =  {v}
                    break
                else:
                    NSG[p].add(v)
                    break
    return NSG

def query_NSG(q,NSG,S,navigating_node):
    """
    Queries NSG graph using Greedy DFS search.

    Required Arguments:
    q: query
    NSG: NSG graph obtained from strongly_connect_NSG()
    S: shard
    navigating_node: Node to start dfs (root)
    """
    visited = set()
    current_idx = navigating_node
    dfs_stack = [(S[current_idx],current_idx)]    
    
    cur_min = 1000000
    cur_min_idx = -1
    
    while (dfs_stack and len(visited) < len(S)):
        v,current_idx = dfs_stack.pop()
                
        dist = get_euclid_dist(q,v)

        visited.add(current_idx)

        prev_min = cur_min
        
        cur_min_idx = cur_min_idx if (np.argmin([cur_min,dist]) == 0) else current_idx
        cur_min = min([cur_min,dist])
        
        if (cur_min==0.0 or cur_min-prev_min > 0):
            break
        
        try:
            adj = NSG[current_idx]
        except:
            break

        for n in adj:
            if (n not in visited):
                dfs_stack.append((S[n],n))
    return cur_min,cur_min_idx

def save_NSG(NSG,nsgdir,nsgfilename):
    with open(os.path.join(nsgdir,nsgfilename),'wb') as fh:
        pickle.dump(NSG,fh)

def load_NSG(nsgdir,nsgfilename):
    fh = open(os.path.join(nsgdir,nsgfilename),"rb")
    return pickle.load(fh)