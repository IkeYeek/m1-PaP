#!/usr/bin/env python3
from expTools import *

# MPI configurations to test
mpi_configs = ['"-np 1"', '"-np 2"', '"-np 4"', '"-np 8"', '"-np 16"']

# Common options for all runs
easypapOptions = {
    "-k": ["life"],                 # Kernel: Game of Life
    "-i": [10],                     # Number of iterations
    "-v": ["mpi"],             # Version: MPI+OpenMP
    "-a": ["random"],               # Initialization pattern
    "-s": [8192],                   # Grid size (8192x8192)
    "--label": ["mpi_speedup"],     # Label for grouping results
    "-of": ["life_speedup.csv"],    # Output file
    "-mpi": mpi_configs,            # MPI configurations to test
}

# OpenMP configurations (with fixed thread count since we're focusing on MPI scaling)
ompICV = {
    "OMP_SCHEDULE": ["static"],     # Static scheduling
    "OMP_PLACES": ["threads"],      # Thread placement
    "OMP_NUM_THREADS": [1]          # Only 1 thread per process to isolate MPI scaling
}

nbruns = 3  # Number of runs per configuration for more stable results

# Execute MPI+OpenMP versions
execute("./run", ompICV, easypapOptions, nbruns, verbose=False, easyPath=".")

# Execute sequential version for baseline comparison
easypapOptions["-v"] = ["seq"]
easypapOptions["-mpi"] = ['"-np 1"']  # Only need 1 process for sequential
seqOmpICV = {"OMP_NUM_THREADS": [1]}
execute("./run", seqOmpICV, easypapOptions, nbruns, verbose=False, easyPath=".")

print("Experiments complete. Recommended plots:")
print("plots/easyplot.py -if life_speedup.csv -v mpi -- x=mpi_np y=speedup")
print("This will show the speedup relative to the sequential version for different MPI process counts.")