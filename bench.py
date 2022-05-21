import matplotlib.pyplot as plt
from cupyx.profiler import benchmark
from timers import time_func
from implementations import *

repeat = 10 # num of bench repeats to retrieve time stats

size = [10,50,100,500,1000,1500,2000]
num_operations = [1,5,10,50,100,200,300]


# initialize dictionaries to store mean values of benchmarks
bench_size = {}
bench_oper = {}
bench_alloc = {}

for func in ['gpu_copy', 'unified']:
    bench_size[func] = {}
    bench_size[func]['cpu'] = []
    bench_size[func]['gpu'] = []

    bench_oper[func] = {}
    bench_oper[func]['cpu'] = []
    bench_oper[func]['gpu'] = []

bench_alloc['size'] = {}
bench_alloc['size']['cpu'] = []
bench_alloc['size']['gpu'] = []


# bench for different sizes
num_arrays = 3
num_oper = num_operations[3] 

for N in size:

    arrays, t_cpu, t_gpu = time_func(unified_arrays,N,num_arrays)
    bench_alloc['size']['cpu'].append(t_cpu)
    bench_alloc['size']['gpu'].append(t_gpu)

    bench_cp = benchmark(func=gpu_copy,args=(N, num_oper),n_repeat=repeat)
    bench_un = benchmark(func=unified,args=((arrays,num_oper)),n_repeat=repeat)

    bench_size['gpu_copy']['cpu'].append(bench_cp.cpu_times.mean()*10**3) # append cpu time in msecs
    bench_size['gpu_copy']['gpu'].append(bench_cp.gpu_times.mean()*10**3)

    bench_size['unified']['cpu'].append(bench_un.cpu_times.mean()*10**3) 
    bench_size['unified']['gpu'].append(bench_un.gpu_times.mean()*10**3)


# plot for different sizes
for dev in ['cpu', 'gpu']:

    fig, ax = plt.subplots()
    ax.set(xlabel = "Array size ( NxN )", ylabel = f"Time elapsed in {dev.upper()} (msec)",
        title = f"{dev.upper()} time comparison for different array sizes\n Operations Repeat Number: {num_oper}")

    for func in ['gpu_copy', 'unified']:
        ax.plot(size, bench_size[func][dev], label = f"{func}", marker='x')
        
    ax.plot(size, bench_alloc['size'][dev], label = f"unified_alloc", marker='o')

    ax.legend(loc='upper left')
    ax.grid(True)
    figname = f"time_size_{dev}.png"
    fig.savefig(figname, bbox_inches="tight")



# bench for different operation number
num_arrays = 3
N = 500

for num_oper in num_operations:

    arrays = unified_arrays(N,num_arrays)

    bench_cp = benchmark(func=gpu_copy,args=(N, num_oper),n_repeat=repeat)
    bench_un = benchmark(func=unified,args=((arrays,num_oper)),n_repeat=repeat)

    bench_oper['gpu_copy']['cpu'].append(bench_cp.cpu_times.mean()*10**3) # append cpu time in msecs
    bench_oper['gpu_copy']['gpu'].append(bench_cp.gpu_times.mean()*10**3)

    bench_oper['unified']['cpu'].append(bench_un.cpu_times.mean()*10**3) 
    bench_oper['unified']['gpu'].append(bench_un.gpu_times.mean()*10**3)


# plot for different operation repeat number
for dev in ['cpu', 'gpu']:

    fig, ax = plt.subplots()
    ax.set(xlabel = "Operations Repeat Number", ylabel = f"Time elapsed in {dev.upper()} (msec)",
        title = f"{dev.upper()} time comparison for different operations repeat number\n Array size: {N}")

    for func in ['gpu_copy', 'unified']:
        ax.plot(num_operations, bench_oper[func][dev], label = f"{func}", marker='x')

    ax.legend(loc='upper left')
    ax.grid(True)
    figname = f"time_oper_{dev}.png"
    fig.savefig(figname, bbox_inches="tight")