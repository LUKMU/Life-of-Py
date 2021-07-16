import sys
import numpy as np
import math
from numba import cuda
from numba import vectorize

def initialize(grid, numalivecells, f):
	i = 1
	while (i <= numalivecells):
		text = f.readline().strip()
		print(text)
		text = text.split(" ")
		x = int(text[0])
		y = int(text[1])
		print("(" + str(x) + "," + str(y) + ")")
		grid[x][y] = 1
		i += 1
	print(grid)

	
#cuda kernel function
@cuda.jit
def gol_kernel(oldgrid, newgrid, numrows, numcols):
	x, y = cuda.grid(2)
	if x < (oldgrid.shape[0]-1) and y < (oldgrid.shape[1]-1) and x > 0 and y > 0:
		#creates ring around new grid based on old grid's corners and sides
		if x == 1 and y == 1:
			newgrid[numrows-1][numcols-1] = oldgrid[1][1]
		if x == (numrows-2) and y == (numcols-2):
			newgrid[0][0] = oldgrid[numrows-2][numcols-2]
		if x == 1 and y == (numcols - 2):
			newgrid[numrows-1][0] = oldgrid[1][numcols-2]
		if x == (numrows-2) and y == 1:
			newgrid[0][numcols-1] = oldgrid[numrows-2][1]
		if y == 1:
			newgrid[x][numcols-1] = oldgrid[x][1]
		if y == (numcols - 2):
			newgrid[x][0] = oldgrid[x][numcols - 2]
		if x == (numrows - 2):
			newgrid[0][y] = oldgrid[numrows - 2][y]
		if x == 1:
			newgrid[numrows-1][y] = oldgrid[1][y]

		#rules regulating cell life and death are enforced

		numaliveneighbors = 0
		if oldgrid[x-1][y-1] == 1:
			numaliveneighbors += 1
		if oldgrid[x-1][y] == 1:
			numaliveneighbors += 1
		if oldgrid[x-1][y+1] == 1:
			numaliveneighbors += 1
		if oldgrid[x][y-1] == 1:
			numaliveneighbors += 1
		if oldgrid[x][y+1] == 1:
			numaliveneighbors += 1
		if oldgrid[x+1][y-1] == 1:
			numaliveneighbors += 1
		if oldgrid[x+1][y] == 1:
			numaliveneighbors += 1
		if oldgrid[x+1][y+1] == 1:
			numaliveneighbors += 1

		if oldgrid[x][y] == 0:
			if numaliveneighbors == 3:
				newgrid[x][y] = 1
			else:
				newgrid[x][y] = 0
		if oldgrid[x][y] == 1:
			if numaliveneighbors == 2 or numaliveneighbors == 3:
				newgrid[x][y] = 1
			else:
				newgrid[x][y] = 0		

#name of input file, number of iterations for GOL to go through, output file name (if needed), 
#and option to determine whether program will display generations or print them to a file
infile = sys.argv[1]
iters = int(sys.argv[2])
outfile = sys.argv[3]
disp = sys.argv[4]

f = open(infile, 'r')

text = f.readline().strip()
text = text.split(" ")
numrows = int(text[0])
numcols = int(text[1])
numalivecells = int(text[2])
i = 1
numrows += 2
numcols += 2

grid1 = np.zeros((numrows, numcols), dtype = np.int32)
grid2 = np.zeros_like(grid1)

#creates initial grid

initialize(grid1, numalivecells, f)

#call to cuda kernel

threadsperblock = (numrows, numcols)
blockspergrid_x = math.ceil(grid1.shape[0] / threadsperblock[0])
blockspergrid_y = math.ceil(grid1.shape[1] / threadsperblock[1])
blockspergrid = (blockspergrid_x, blockspergrid_y)

for x in range(iters):
	if (x % 2) == 0: 
		gol_kernel[blockspergrid, threadsperblock](grid1, grid2, numrows, numcols)
	else:
		gol_kernel[blockspergrid, threadsperblock](grid2, grid1, numrows, numcols)		
	cuda.synchronize()
	if (x % 2) == 0:
		print(grid2)
	else:
		print(grid1)
	print(' ')
