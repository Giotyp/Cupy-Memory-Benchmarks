import argparse
from implementations import *
from cupyx.profiler import benchmark

parser = argparse.ArgumentParser(description='Test different cupy memory tactics')
parser.add_argument('-size', metavar='N', type=int, required=True, help='Array size used for testing')
parser.add_argument('-repeat', type=int, required=True, help='Benchmark repetition times')
parser.add_argument('-gpu_copy', action='store_true', help='Initialize to cpu and copy to gpu')
parser.add_argument('-unified', action='store_true', help='Use cupy unified memory')

args = parser.parse_args()

N = args.size

if args.gpu_copy:
    a_cpu = np.random.rand(N,N).astype(np.dtype('float32'))
    b_cpu = np.random.rand(N,N).astype(np.dtype('float32'))
    print(benchmark(func=gpu_copy,args=(a_cpu, b_cpu),n_repeat=args.repeat, n_warmup=2))

if args.unified:
    arrays = unified_arrays(N,3)
    print(benchmark(func=unified,args=((arrays,)),n_repeat=args.repeat, n_warmup=2))
