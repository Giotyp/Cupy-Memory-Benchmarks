import numpy as np
import cupy as cp
from sys import getsizeof

def cpu_init(N):
    a = np.random.rand(N,N).astype('float32')
    b = np.random.rand(N,N).astype('float32')
    c = np.random.rand(N,N).astype('float32')
    a_gpu = cp.asarray(a)
    b_gpu = cp.asarray(b)
    res = cp.dot(a_gpu, b_gpu)
    d = cp.asnumpy(res)
    np.dot(c,d) 
    
def gpu_init(N):
    a_gpu = cp.random.randn(N,N,dtype='float32')
    b_gpu = cp.random.randn(N,N,dtype='float32')
    c = np.random.rand(N,N).astype('float32')
    res = cp.dot(a_gpu, b_gpu)
    d = cp.asnumpy(res)
    np.dot(c,d) 
    #print(res)

def pool_stats(mempool):
    print('used:',mempool.used_bytes(),'bytes')
    print('total:',mempool.total_bytes(),'bytes\n')

def unified_init(N):
    dt = cp.dtype('float32').itemsize
    size = N*N*dt
    
    ptr_a = cp.cuda.malloc_managed(size)
    ptr_b = cp.cuda.malloc_managed(size)
    return ptr_a, ptr_b

def unified(N,ptr_a, ptr_b):
    shape = (N,N)
    a = cp.ndarray(shape=shape, dtype=cp.float32, memptr=ptr_a)
    for i in range(N):
        a[i] = cp.random.randn(N,dtype='float32')
    
    
    b = cp.ndarray(shape=shape, dtype=cp.float32, memptr=ptr_b)
    for i in range(N):
        b[i] = cp.random.randn(N,dtype='float32')

    c = np.random.rand(N,N).astype('float32')

    res = cp.dot(a, b)
    d = cp.asnumpy(res)
    cp.dot(c,d) 
