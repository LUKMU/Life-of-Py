#!/usr/bin/env python

import sys
import numpy as np
import math
from mpi4py import MPI
#from numba import cuda
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
#from numba import cuda

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
sendarr = np.zeros(numcols)

grid1 = np.zeros((numrows, numcols), dtype = np.int32)
grid2 = np.zeros_like(grid1)
#creates initial grid
initialize(grid1, numalivecells, f)

if rank == 0:
	print("RANK 0")
	sendarr = grid1[rank]
	send = comm.isend(sendarr, dest=(numrows-1), tag=rank)
	send.wait()

	send = comm.isend(sendarr, dest=(rank+1), tag=rank)
	send.wait()

	top = comm.irecv(source=(numrows-1), tag=(numrows-1))
	datatop = top.wait()

	btm = comm.irecv(source=(rank+1), tag=(rank+1))
	databtm = btm.wait()

elif rank == (numrows - 1):
	print("RANK" + str(rank))

	sendarr = grid1[rank]
	send = comm.isend(sendarr, dest=0, tag=rank)
	send.wait()

	send = comm.isend(sendarr, dest=(rank-1), tag=rank)
	send.wait()
	
	top = comm.irecv(source=(rank-1), tag=(rank-1))
	datatop = top.wait()

	btm = comm.irecv(source=0, tag=0)
	databtm = btm.wait()
else:

	print("HI")

	sendarr = grid1[rank]
	send = comm.isend(sendarr, dest=rank-1, tag=rank)
	send.wait()

	send = comm.isend(sendarr, dest=rank+1, tag=rank)
	send.wait()

	top = comm.irecv(source=(rank-1), tag=(rank-1))
	datatop = top.wait()

	btm = comm.irecv(source=(rank+1), tag=(rank+1))
	databtm = btm.wait()

	print(databtm)
