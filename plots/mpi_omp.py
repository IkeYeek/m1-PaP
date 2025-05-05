#!/usr/bin/env python3
from expTools import *

# Stable process counts to test
mpi_configs = ['"-np 1"', '"-np 4"', '"-np 8"']  # Adjust based on your system

# Common options for all runs
easypapOptions = {
    "-k": ["life"],                 # Kernel: Game of Life
    "-i": [11],                     # Number of iterations
    "-v": ["mpi", "mpi_omp"],       # Both versions to compare
    "-a": ["random"],               # Initialization pattern
    "-s": [8192],                   # Grid size
    "--label": ["version_compare"], # Label for grouping results
    "-of": ["life_compare.csv"],    # Output file
    "-mpi": mpi_configs,            # MPI configurations
    "-sh": [""]                     # Shared memory flag
}

# OpenMP configurations
# For mpi version: 1 thread (pure MPI)
# For mpi_omp version: test different thread counts
ompICV = {
    "OMP_SCHEDULE": ["static"],
    "OMP_PLACES": ["threads"],
    # Will be set differently for each version in execute()
}

nbruns = 3  # Number of runs per configuration

# First run pure MPI version (OMP_NUM_THREADS=1)
current_ompICV = ompICV.copy()
current_ompICV["OMP_NUM_THREADS"] = [1]
easypapOptions["-v"] = ["mpi"]
execute("./run", current_ompICV, easypapOptions, nbruns, verbose=False, easyPath=".")

# Then run hybrid MPI+OpenMP version with different thread counts
current_ompICV["OMP_NUM_THREADS"] = [1, 2, 4]  # Test different thread counts
easypapOptions["-v"] = ["mpi_omp"]
execute("./run", current_ompICV, easypapOptions, nbruns, verbose=False, easyPath=".")

# Sequential version for baseline
easypapOptions["-v"] = ["seq"]
easypapOptions["-mpi"] = ['"-np 1"']
seqOmpICV = {"OMP_NUM_THREADS": [1], "OMP_PLACES": ["threads"]}
execute("./run", seqOmpICV, easypapOptions, nbruns, verbose=False, easyPath=".")

print("\nExperiments complete. Recommended plots:")
print("1. Version comparison (fixed process count):")
print("   ./plots/easyplot.py -if life_compare.csv -v mpi,mpi_omp -- col=mpi_np row=OMP_NUM_THREADS y=Gcell/s")
print("2. Speedup comparison:")
print("   ./plots/easyplot.py -if life_compare.csv -v mpi,mpi_omp -- col=mpi_np row=OMP_NUM_THREADS y=speedup")
print("3. Efficiency comparison:")
print("   ./plots/easyplot.py -if life_compare.csv -v mpi,mpi_omp -- col=mpi_np row=OMP_NUM_THREADS y=efficiency")