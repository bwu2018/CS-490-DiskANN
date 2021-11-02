import numpy as np
import sys
import os
import glob
import time
import subprocess

N_SHARDS = 3
PATH_TO_SHARDS = '../sift/shards/'
PATH_TO_NSG = '../nsg-imp/'

L = 40
R = 50
C = 500

logfile = open('nsg_out.txt', 'w')

args = [int(arg) for arg in sys.argv[1:]]
if len(args) == 1:
    N_SHARDS = args[0]
elif len(args) == 5:
    L, R, C = args
elif len(args) == 6:
    N_SHARDS, L, R, C = args

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

    # delete existing indexes    
    files = glob.glob(PATH_TO_SHARDS + 'nsg_indexes/*')
    for f in files:
        os.remove(f)

    os.chdir(PATH_TO_NSG)

    for shard_num in range(1, N_SHARDS + 1):
        print_and_log('shard: ' + str(shard_num))
        shard_args = []
        shard_args.append('./build/tests/test_nsg_index')
        shard_args.append(PATH_TO_SHARDS + 'sift_shard' + str(shard_num) + '.fvecs')
        shard_args.append(PATH_TO_SHARDS + 'graphs/sift_shard' + str(shard_num) + '.graph')
        shard_args.append(L)
        shard_args.append(R)
        shard_args.append(C)
        shard_args.append(PATH_TO_SHARDS + 'nsg_indexes/sift_shard' + str(shard_num) + '.nsg')
        run_command(shard_args)
    
    print_and_log('Total time taken: ' + str(time.time() - start_time))


if __name__ == '__main__':
    main()
