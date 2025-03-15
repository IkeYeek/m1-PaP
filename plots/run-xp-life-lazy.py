#!/usr/bin/env python3
from expTools import *

# Test different tile sizes for lazy evaluation
easypapOptions = {
    "-k": ["life"],
    "-i": [100],
    "-v": ["omp_lazy"],
    "-s": [4096],
    "-tw": [8, 16, 32, 64, 128, 256],
    "-th": [8, 16, 32, 64, 128, 256],
    "-of": ["lazy-tile-sizes.csv"],
    "-wt": ["opt"]
}

ompICV = {
    "OMP_SCHEDULE": ["dynamic"],
    "OMP_PLACES": ["threads"],
    "OMP_NUM_THREADS": [20]
}

nbruns = 3
execute("./run ", ompICV, easypapOptions, nbruns, verbose=False, easyPath=".")

print("Recommended plot:")
print(" plots/easyplot.py -if lazy-tile-sizes.csv -3d -v omp_lazy -- col=tile_width row=tile_height val=iterations")