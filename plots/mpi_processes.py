#!/usr/bin/env python3
from expTools import *

# Get the sequential reference time first (important to establish baseline)
seq_options = {
    "-k": ["life"],           # Kernel: Game of Life
    "-i": [10],               # Number of iterations
    "-v": ["seq"],            # Version: Sequential
    "-a": ["random"],         # Initialization pattern
    "-s": [8192],             # Grid size (8192x8192)
    "--label": ["seq_ref"],   # Label for reference
    "-of": ["life_speedup.csv"] # Output file
}

seq_omp = {
    "OMP_PLACES=threads"
    "OMP_NUM_THREADS": [1]
}

# Run sequential version once for reference
execute("./run", seq_omp, seq_options, 3, verbose=True, easyPath=".")

# Now run MPI configurations
mpi_process_counts = [1, 2, 4, 8, 16]

# Common options for all MPI runs
mpi_options = {
    "-k": ["life"],           # Kernel: Game of Life
    "-i": [10],               # Number of iterations
    "-v": ["mpi"],            # Version: MPI
    "-a": ["random"],         # Initialization pattern
    "-s": [8192],             # Grid size (8192x8192)
    "--label": ["mpi_speedup"], # Label for grouping results
    "-of": ["life_speedup.csv"] # Output file
}

# Create separate runs for each MPI configuration
for np in mpi_process_counts:
    current_options = mpi_options.copy()
    current_options["-mpi"] = [f'"-np {np}"']
    
    # For larger runs, may need a hostfile
    if np > 8:
        current_options["-mpi"] = [f'"-np {np}"']
    
    # Environment variables
    mpi_omp = {
        "OMP_SCHEDULE": ["static"],
        "OMP_PLACES": ["threads"],
        "OMP_NUM_THREADS": [1]  # Only 1 thread per process to isolate MPI scaling
    }
    
    execute("./run", mpi_omp, current_options, 3, verbose=True, easyPath=".")

print("Experiments complete. Recommended plots:")
print("plots/easyplot.py -if life_speedup.csv -g mpi_np -- x=mpi_np y=speedup")