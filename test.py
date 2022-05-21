import argparse
import sys
from implementations import *
from cupyx.profiler import benchmark
from timers import time, time_func

parser = argparse.ArgumentParser(description='Test different cupy memory tactics')
parser.add_argument('-size', type=int, help='Array size (NxN) used for testing (default=10)')
parser.add_argument('-repeat', type=int, help='Benchmark repetition times (default=10)')
parser.add_argument('-num_oper', type=int, help='Perform #num_oper operation loops (default=10) ')
parser.add_argument('-gpu_copy', action='store_true', help='Memory copies between CPU - GPU')
parser.add_argument('-unified', action='store_true', help='Use cupy unified memory')

if len(sys.argv) == 1:
    parser.print_help()
    exit(0)


args = parser.parse_args()

N = args.size
repeat = args.repeat
num_oper = args.num_oper

if N is None: N = 10
if repeat is None: repeat = 10
if num_oper is None: num_oper = 10 

print(f"Benchmarks ran {repeat} times for arrays {N}x{N} and {num_oper} operations loop\n")

if args.gpu_copy:
    print(benchmark(func=gpu_copy,args=(N, num_oper),n_repeat=repeat))

if args.unified:
    num_arrays = 3
    arrays, t_cpu, t_gpu = time_func(unified_arrays,N,num_arrays)
    print( "{}      :    CPU: {:.3f}us       GPU-0: {:.3f}us".format("unified_arrays", t_cpu*10**3, t_gpu*10**3))
    print(benchmark(func=unified,args=((arrays,num_oper)),n_repeat=repeat, n_warmup=2))