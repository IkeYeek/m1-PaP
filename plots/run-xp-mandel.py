#!/usr/bin/env python3
from expTools import *

# Recommended Plot:
# plots/easyplot.py -if mandel.csv -v omp_tiled -- col=schedule row=label

easypapOptions = {
    "-k": ["mandel"],
    "-i": [5],
    "-v": ["omp_tiled"],
    "-s": [512],
    "-ts": [8, 16, 32],
    "--label": ["square"],
    "-of": ["mandel.csv"],
}

# OMP Internal Control Variable
ompICV = {
    "OMP_SCHEDULE": ["static", "static,1", "dynamic"],
    "OMP_NUM_THREADS": [1] + list(range(4, os.cpu_count() + 1, 4)),
}

nbruns = 1
# Lancement des experiences
execute("./run ", ompICV, easypapOptions, nbruns, verbose=False, easyPath=".")

del easypapOptions["-ts"]
easypapOptions["--label"] = ["line"]
easypapOptions["-th"] = [1]
easypapOptions["-tw"] = [64, 256, 512]

execute("./run ", ompICV, easypapOptions, nbruns, verbose=False, easyPath=".")

# Lancement de la version seq avec le nombre de thread impose a 1

easypapOptions = {
    "-k": ["mandel"],
    "-i": [5],
    "-v": ["seq"],
    "-s": [512],
    "-of": ["mandel.csv"],
}
ompICV = {"OMP_NUM_THREADS": [1]}
execute("./run ", ompICV, easypapOptions, nbruns, verbose=False, easyPath=".")


print("Recommended plot:")
print(" plots/easyplot.py -if mandel.csv -v omp_tiled -- col=schedule row=label")