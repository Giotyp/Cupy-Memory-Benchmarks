import numpy as np
import cupy as cp
import cupyx.scipy.fft as cufft
from sys import getsizeof
import ctypes

def gpu_copy(N, num_operations):
    '''
    Initialize a_cpu and b_cpu with shape (N,N)
    Perform #num_operation operations loops 
    Copy arrays from gpu to cpu and reverse
    '''
    a_cpu = np.random.rand(N,N).astype(np.dtype('float32'))
    b_cpu = np.random.rand(N,N).astype(np.dtype('float32'))

    for i in range(num_operations):
        a_gpu = cp.asarray(a_cpu)
        b_gpu = cp.asarray(b_cpu)

        c_gpu = cp.dot(a_gpu, b_gpu)
        a_gpu = cufft.fft(c_gpu)
        b_gpu = a_gpu + c_gpu

        a_cpu = cp.asnumpy(a_gpu)
        b_cpu = cp.asnumpy(b_gpu)
        c_cpu = cp.asnumpy(c_gpu)

        c_cpu = b_cpu + a_cpu
        a_cpu = np.dot(b_cpu, c_cpu)
        b_cpu = a_cpu + c_cpu

def unified_pointers(N, elmnts_type, num_pointers = 1):
    '''
    Return num_pointers  MemoryPointer instances to N bytes 
    unified memory of elmnts_type 
    '''
    dt = elmnts_type.itemsize
    size = N*N*dt
    
    pointers = []
    for i in range(num_pointers):
        ptr = cp.cuda.malloc_managed(size)
        pointers.append(ptr)
    return pointers

def unified_arrays(N,num_arrays):
    '''
    Return [[array_cpu, array_gpu],...] with arrays (N,N) 
    from retrieved pointers to Unified Memory
    '''
    shape = (N,)

    # Get array pointers
    pointers = unified_pointers(N,np.dtype('float32'),num_arrays)

    arrays = []
    for ptr in pointers:
        # convert np.float32 to ctype_float
        ctp = np.ctypeslib.as_ctypes_type(np.dtype('float32'))   
        # cast pointers value to ctype POINTER of c_float  
        pt = ctypes.cast(ptr.ptr,ctypes.POINTER(ctp))
        # create numpy array from POINTER
        x_cpu = np.ctypeslib.as_array(pt, shape)
        # create cupy array from pointer ptr
        x_gpu = cp.ndarray(shape=shape, dtype=np.dtype('float32'), memptr=ptr)

        arrays.append([x_cpu, x_gpu])
    return arrays

def unified(arrays, num_operations):
    '''
    Take arrays arr_x = [x_cpu/x_gpu] in unified memory
    where x = a, b, c from arrays=[arr_a, arr_b, arr_c]
    Initialize a_cpu and b_cpu with shape (N,N)
    Perform #num_operation operations loops
    Use unified memory arrays instead of copying
    '''
    dv = cp.cuda.Device(0)

    arr_a, arr_b, arr_c = arrays

    a_cpu, a_gpu = arr_a
    b_cpu, b_gpu = arr_b
    c_cpu, c_gpu = arr_c

    N = a_cpu.shape[0]
    a_cpu = np.random.rand(N,N).astype(np.dtype('float32'))
    b_cpu = np.random.rand(N,N).astype(np.dtype('float32'))

    for i in range(num_operations):
        a_gpu = cp.asarray(a_cpu)
        b_gpu = cp.asarray(b_cpu)

        c_gpu = cp.dot(a_gpu, b_gpu)
        a_gpu = cufft.fft(c_gpu)
        b_gpu = a_gpu + c_gpu

        dv.synchronize() # synchronize device - calls cudaDeviceSynchronize()

        c_cpu = b_cpu + a_cpu
        a_cpu = np.dot(b_cpu, c_cpu)
        b_cpu = a_cpu + c_cpu

