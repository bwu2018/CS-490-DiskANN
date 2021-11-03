import numpy as np
import sys
import os
import glob
import time
import subprocess

N_SHARDS = 40
PATH_TO_SHARDS = '../sift/shards/'
PATH_TO_EFANNA = '../efanna-imp/'

K = 200
L = 200
ITERATIONS = 10
S = 10
R = 100

logfile = open('efanna_out.txt', 'w')

args = [int(arg) for arg in sys.argv[1:]]
if len(args) == 1:
    N_SHARDS = args[0]
elif len(args) == 5:
    K, L, ITERATIONS, S, R = args
elif len(args) == 6:    
    N_SHARDS, K, L, ITERATIONS, S, R = args

def run_command(command_args):
    command_args = [str(cmd) for cmd in command_args]
    command = ' '.join(command_args)
    res = subprocess.Popen(command_args, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    stdout, stderr = res.communicate()
    print_and_log(stdout.decode('UTF-8'))
    print_and_log('`{}` ran with exit code {}'.format(command, res.returncode))

def print_and_log(msg):
    print(msg)
    logfile.write(msg + '\n')    

def main():
    start_time = time.time()
    # delete existing graphs    
    files = glob.glob(PATH_TO_SHARDS + 'graphs/*')
    for f in files:
        os.remove(f)

    os.chdir(PATH_TO_EFANNA)

    for shard_num in range(1, N_SHARDS + 1):
        print_and_log('shard: ' + str(shard_num))
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

    
    print_and_log('Total time taken: ' + str(time.time() - start_time))
    
if __name__ == '__main__':
    main()
