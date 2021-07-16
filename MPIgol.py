import sys
import numpy as np
import math
from numba import cuda
from mpi4py import MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()

#function that initializes first grid based on
#input file
def initialize(grid, numalivecells, f):
	i = 1
	while (i <= numalivecells):
		text = f.readline().strip()
		text = text.split(" ")
		x = int(text[0])
		y = int(text[1])
		grid[x][y] = 1
		i += 1
	print(grid)

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
if rank == 0:
	#creates initial grid
	initialize(grid1, numalivecells, f)
	print("RANK 0")

elif rank == (numrows - 1):
	print("RANK " + str(rank))

else:
	print("HI")
