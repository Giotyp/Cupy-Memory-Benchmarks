import numpy as np
import cupy as cp
from sys import getsizeof
import ctypes

def gpu_copy(a_cpu, b_cpu):
    '''
    Take arrays a_cpu and b_cpu
    Copy them to gpu and multiply them
    Copy result back to cpu and increment its elements
    '''

    a_gpu = cp.asarray(a_cpu)
    b_gpu = cp.asarray(b_cpu)
    res = cp.dot(a_gpu, b_gpu)
    
    c = cp.asnumpy(res)
    c[::] += 1

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
    shape = (N,N)

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

        x_cpu[::] = np.random.rand(N,N).astype(np.dtype('float32')) 

        arrays.append([x_cpu, x_gpu])
    return arrays

def unified(arrays):
    '''
    Take arrays arr_x = [x_cpu/x_gpu] in unified memory
    where x = a, b, c from arrays=[arr_a, arr_b, arr_c]
    Multiply them in gpu and get result also in UM
    Increment elements of result in cpu
    '''
    dv = cp.cuda.Device(0)

    arr_a, arr_b, arr_c = arrays

    a_gpu = arr_a[1]
    b_gpu = arr_b[1]
    c_cpu, c_gpu = arr_c

    c_gpu[::] = cp.dot(a_gpu, b_gpu) #result  in managed memory
    dv.synchronize()

    c_cpu[::] += 1 

