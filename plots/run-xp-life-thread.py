#!/usr/bin/env python3
from expTools import *

# Test scalability of parallel lazy evaluation
easypapOptions = {
    "-k": ["life"],
    "-i": [100],
    "-v": ["omp_lazy"],
    "-s": [4096],
    "-of": ["lazy-scalability.csv"],
    "-wt": ["opt"]
}

# Test with different thread counts
ompICV = {
    "OMP_SCHEDULE": ["dynamic"],
    "OMP_PLACES": ["threads"],
    "OMP_NUM_THREADS": [1, 2, 4, 8, 16, 20, 24, 32, 40, 47]
}

nbruns = 3
execute("./run ", ompICV, easypapOptions, nbruns, verbose=False, easyPath=".")

# Add sequential version for speedup reference
easypapOptions["-v"] = ["seq"]
ompICV["OMP_NUM_THREADS"] = [1]
execute("./run ", ompICV, easypapOptions, nbruns, verbose=False, easyPath=".")

print("Recommended plot:")
print(" plots/easyplot.py -if lazy-scalability.csv -v omp_lazy -- col=num_threads row=size")