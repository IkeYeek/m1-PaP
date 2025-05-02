#!/usr/bin/env python3
from expTools import *

# Configuration for MPI+OpenMP Life experiments
easypapOptions = {
    "-k": ["life"],                  # Kernel: life
    "-i": [5],                       # Number of iterations
    "-v": ["mpi_omp"],               # Version: MPI with OpenMP
    "-a": ["moultdiehard130"],       # Initial configuration
    "-d": ["M"],                     # Display mode
    "-s": [512],                     # Domain size
    "--label": ["square"],           # Label
    "-of": ["life_mpi.csv"],         # Output file
}

# MPI configuration with different process counts
# Note: the -mpi option takes a quoted string value
mpi_configs = ["-np 1", "-np 2", "-np 4", "-np 8"]

nbruns = 1  # Number of runs per configuration

# First run MPI+OMP with OMP_NUM_THREADS=1 for all MPI configurations as baseline
baselineOmpICV = {
    "OMP_SCHEDULE": ["static,1"],
    "OMP_PLACES": ["threads"],
    "OMP_NUM_THREADS": [1],  # Baseline with 1 thread
}

print("Running baseline experiments with OMP_NUM_THREADS=1...")
for mpi_config in mpi_configs:
    easypapOptions["-mpi"] = [f'"{mpi_config}"']  # Properly quoted string
    print(f"Running baseline with {mpi_config}...")
    execute("./run", baselineOmpICV, easypapOptions, nbruns, verbose=False, easyPath=".")

# Then run with different thread counts
ompICV = {
    "OMP_PLACES": ["threads"],
    "OMP_NUM_THREADS": [2, 4, 8, 10, 16],  # Test various thread counts
}

print("Running scaling experiments with varied thread counts...")
for mpi_config in mpi_configs:
    easypapOptions["-mpi"] = [f'"{mpi_config}"']  # Properly quoted string
    print(f"Running scaling with {mpi_config}...")
    execute("./run", ompICV, easypapOptions, nbruns, verbose=False, easyPath=".")

# Run sequential version for absolute baseline comparison
print("Running sequential version for baseline...")
seqOptions = {
    "-k": ["life"],
    "-i": [5],
    "-v": ["seq"],
    "-s": [512],
    "-of": ["life_mpi.csv"],  # Append to the same file for comparison
}
seqOmpICV = {"OMP_NUM_THREADS": [1]}
execute("./run", seqOmpICV, seqOptions, nbruns, verbose=False, easyPath=".")

print("Experiments complete. Recommended plots:")
print("plots/easyplot.py -if life_mpi.csv -v mpi_omp -- col=schedule row=label")
