import numpy as np
import sys
import os
import glob

N_SHARDS = 3
PATH_TO_SHARDS = '../sift/shards/'
PATH_TO_EFANNA = '../efanna-imp/'

K = 50
L = 70
ITERATIONS = 10
S = 10
R = 10

args = sys.argv[1:]
if len(args) == 1:
    N_SHARDS = args[0]
elif len(args) == 5:
    K = args[0]
    L = args[1]
    ITERATIONS = args[2]
    S = args[3]
    R = args[4]
elif len(args) == 6:
    N_SHARDS = args[0]
    K = args[1]
    L = args[2]
    ITERATIONS = args[3]
    S = args[4]
    R = args[5]

def run_command(command_args):
    command = ' '.join([str(cmd) for cmd in command_args])
    res = os.system(command)
    print('`{}` ran with exit code {}'.format(command, res))

# delete existing graphs    
files = glob.glob(PATH_TO_SHARDS + 'graphs/*')
for f in files:
    os.remove(f)

os.chdir(PATH_TO_EFANNA)
run_command(['ls'])
for shard_num in range(1, N_SHARDS + 1):
    print('shard:', shard_num)
    shard_args = []
    shard_args.append('./tests/test_nndescent')
    shard_args.append(PATH_TO_SHARDS + 'sift_shard' + str(shard_num) + '.fvecs')
    shard_args.append(PATH_TO_SHARDS + 'graphs/sift_shard' + str(shard_num) + '.graph')
    shard_args.append(K)
    shard_args.append(L)
    shard_args.append(ITERATIONS)
    shard_args.append(S)
    shard_args.append(R)
    run_command(shard_args)
