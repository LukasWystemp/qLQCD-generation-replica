### Script to generate quenched Wilson gauge fields, with the option of tadpole- and Symanzik-improvement.
### Modified to include multiprocessing based on Panagiotis's script.
from __future__ import print_function
import os, sys, string
import numpy as np
from gauge_latticeqcd import *
import lattice_collection as lc
#from multiprocessing import Pool
import functools

### settings
Nt = 6
Nx = 6
Ny = 6
Nz = 6
Ncfg = 10          # number of lattices to generate
action = 'W'    # W = Wilson, Wilson with rectangle improvements, W_T and WR_T = With tadpole improvement
betas = 5.7    # betas to be generated, beta = 6/g^2
startcfg = 0       # warm start (0) or existing cfg number to start the Markov chain
Nhits = 10         # hits between each update
Nmatrix = 10000    # number of random SU(3) matrices to be used for updates
epsilon = 0.2      # how "far" away from identity the updates will be
threads = 1        # threads used in multiprocessing

Nu0_step = 1       # if tadpole improving, number of cfgs to skip between calculating u0.
Nu0_avg = 1        # if tadpole improving, number of u0 values to average together before updating
u0 = 1.            # u0 = <W11>^(1/4); if tadpole improving and continuing from existing lattices, set here.  Else ignore.

### generate lattices


dir_name = action + '_' + str(Nt) + 'x' + str(Nx) + 'x' + str(Ny) + 'x' + str(Nz) + '_b' + str(int(betas * 100))
    
### create output directory if it does not exist
if not os.path.exists(dir_name):
    os.mkdir(dir_name) 
else:
    print("Directory exists for beta ", betas)
    
generate(betas, u0=u0, action=action, Nt=Nt, Nx=Nx, Ny=Ny, Nz=Nz, startcfg=startcfg, Ncfg=Ncfg, Nhits=Nhits, Nmatrix=Nmatrix, epsilon=epsilon, Nu0_step=Nu0_step, Nu0_avg=Nu0_avg)

### initialize multiprocessing
#p = Pool(threads)

### function to be calculated needs to use functools to work with map
#func = functools.partial(generate, u0=u0, action=action, Nt=Nt, Nx=Nx, Ny=Ny, Nz=Nz, startcfg=startcfg, Ncfg=Ncfg, Nhits=Nhits, Nmatrix=Nmatrix, epsilon=epsilon, Nu0_step=Nu0_step, Nu0_avg=Nu0_avg)
#p.map(func, betas) # call multiprocessing map function
#p.terminate()      # terminate multiprocessing
