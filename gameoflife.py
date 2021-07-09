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
			print("BOTTOM RIGHT CORNER")
		if x == (numrows-2) and y == (numcols-2):
			newgrid[0][0] = oldgrid[numrows-2][numcols-2]
			print("TOP LEFT CORNER")
		if x == 1 and y == (numcols - 2):
			newgrid[numrows-1][0] = oldgrid[1][numcols-2]
			print("BOTTOM LEFT CORNER")
		if x == (numrows-2) and y == 1:
			newgrid[0][numcols-1] = oldgrid[numrows-2][1]
			print("TOP RIGHT CORNER")
		if y == 1:
			newgrid[x][numcols-1] = oldgrid[x][1]
			print("RIGHT SIDE")
		if y == (numcols - 2):
			newgrid[x][0] = oldgrid[x][numcols - 2]
			print("LEFT SIDE")
		if x == (numrows - 2):
			newgrid[0][y] = oldgrid[numrows - 2][y]
			print("TOP SIDE")
		if x == 1:
			newgrid[numrows-1][y] = oldgrid[1][y]
			print("BOTTOM SIDE")

		#rules regulating cell life and death are enforced

#name of input file, number of iterations for GOL to go through, output file name (if needed), 
#and option to determine whether program will display generations or print them to a file
infile = sys.argv[1]
iters = sys.argv[2]
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

threadsperblock = (16, 16)
blockspergrid_x = math.ceil(grid1.shape[0] / threadsperblock[0])
blockspergrid_y = math.ceil(grid1.shape[1] / threadsperblock[1])
blockspergrid = (blockspergrid_x, blockspergrid_y)

gol_kernel[blockspergrid, threadsperblock](grid1, grid2, numrows, numcols)
print("HELLO1")
cuda.synchronize()
print(grid2)
