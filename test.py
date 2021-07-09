from numba import cuda
import numpy as np
import math

@cuda.jit
def increment_a_2d_array(an_array):
	x,y = cuda.grid(2)
	if x < an_array.shape[0] and y < an_array.shape[1]:
		an_array[x, y] += 1

a = np.zeros((3,4))
print(a)

threadsperblock = (16, 16)
blockspergrid_x = math.ceil(a.shape[0] / threadsperblock[0])
blockspergrid_y = math.ceil(a.shape[1] / threadsperblock[1])
blockspergrid = (blockspergrid_x, blockspergrid_y)
increment_a_2d_array[blockspergrid, threadsperblock](a)
print(a)
