#!/usr/bin/env python3
from expTools import *

# Recommended Plot:
# plots/easyplot.py -if mandel.csv -v omp_tiled -- col=schedule row=label

easypapOptions = {
    "-k": ["life"],
    "-i": [5],
    "-v": ["lazy_ompfor"],
    "-a": ["moultdiehard130"],
    "-s": [8192],
    "-tw": [1024],
    "-th": [16],
    "-of": ["data/perf/etape-2/life-opt-speedup.csv"],
    "-wt": ["opt"]
}

# OMP Internal Control Variable
ompICV = {
    "OMP_SCHEDULE": ["static,8", "dynamic"],
    "OMP_PLACES": ["threads",],
    "OMP_NUM_THREADS": [26]
}

nbruns = 1
# Lancement des experiences
execute("./run ", ompICV, easypapOptions, nbruns, verbose=False, easyPath=".")

ompICV = {
    "OMP_NUM_THREADS": [1]
}

del easypapOptions["-tw"]
del easypapOptions["-th"]
easypapOptions["-v"] = ["seq"]
execute("./run ", ompICV, easypapOptions, nbruns, verbose=False, easyPath=".")


print("Recommended plot:")
print(" plots/easyplot.py -if life-heat.csv -v omp_tiled -- col=schedule row=label")
