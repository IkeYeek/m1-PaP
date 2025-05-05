#!/usr/bin/env python3
from expTools import *

# Recommended Plot:
# plots/easyplot.py -if mandel.csv -v omp_tiled -- col=schedule row=label
mpi_conf = ['"-np 1"','"-np 2"','"-np 3"','"-np 5"','"-np 6"','"-np 7"','"-np 8"']
easypapOptions = {
    "-k": ["life"],
    "-i": [10],
    "-v": ["mpi_omp"],
    "-a": ["moultdiehard1398"],
    "-s": [4096],
    "-of": ["mpi_threads.csv"],
    "--label" : ["nb_proc"],
    "-mpi": mpi_conf + ['"-hostfile mymachines"']
}

# OMP Internal Control Variable
ompICV = {
    "OMP_SCHEDULE": ["dynamic"],
    "OMP_PLACES": ["threads"],
    "OMP_NUM_THREADS": [25]
}

nbruns = 1
# Lancement des experiences
execute("./run ", ompICV, easypapOptions, nbruns, verbose=True, easyPath=".")

ompICV = {
    "OMP_NUM_THREADS": [1]
}

easypapOptions["-v"] = ["seq"]
execute("./run ", ompICV, easypapOptions, nbruns, verbose=True, easyPath=".")


print("Recommended plot:")
print(" plots/easyplot.py -if mpi_threads.csv -v omp_tiled -- col=schedule row=label")
