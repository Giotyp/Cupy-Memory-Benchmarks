import argparse
from implementations import *
from cupyx.profiler import benchmark

parser = argparse.ArgumentParser(description='Test different cupy memory tactics')
parser.add_argument('-size', metavar='N', type=int, required=True, help='Array size used for testing')
parser.add_argument('-cpu_init', action='store_true', help='Initialize to cpu and copy to gpu')
parser.add_argument('-gpu_init', action='store_true', help='Initialize to gpu')
parser.add_argument('-unified', action='store_true', help='Use cupy unified memory')

args = parser.parse_args()

N = args.size

if args.cpu_init:
    print(benchmark(func=cpu_init,args=(N,),n_repeat=100))

if args.gpu_init:
    print(benchmark(func=gpu_init,args=(N,),n_repeat=100))

if args.unified:
    ptr_a, ptr_b = unified_init(N)
    print(benchmark(func=unified,args=(N,ptr_a,ptr_b),n_repeat=100))