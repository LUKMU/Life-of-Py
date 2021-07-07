import sys
import numpy as np
from numba import cuda

#name of input file, number of iterations for GOL to go through, output file name (if needed), 
#and option to determine whether program will display generations or print them to a file
infile = sys.argv[1]
iters = sys.argv[2]
outfile = sys.argv[3]
disp = sys.argv[4]

f = open(infile, 'r')

text = f.readline().strip()
text = text.split(" ")
numrows = text[0]
numcols = text[1]
numalivecells = text[2]


