import time
import cupy as cp
def time_func(func,*args, n_warmups = 0):
    '''
    Time a given function in cpu and gpu
    Return func's output and result time in msecs
    '''
    
    # Warmup calls 
    for i in range(n_warmups):
        func(*args)

    # declare events and time counters    
    start_gpu = cp.cuda.Event()
    end_gpu = cp.cuda.Event()
    start_gpu.record()
    start_cpu = time.perf_counter()

    ret = func(*args)

    end_cpu = time.perf_counter()
    end_gpu.record()
    end_gpu.synchronize()

    # Calculate time elapsed
    t_gpu_ms = cp.cuda.get_elapsed_time(start_gpu, end_gpu)
    t_cpu_s = end_cpu - start_cpu

    # return time in msecs
    t_cpu_ms = t_cpu_s * 10**3

    return ret, t_cpu_ms, t_gpu_ms