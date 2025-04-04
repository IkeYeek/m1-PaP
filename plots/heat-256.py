#!/usr/bin/env python3
from expTools import *

# Recommended Plot:
# plots/easyplot.py -if mandel.csv -v omp_tiled -- col=schedule row=label

easypapOptions = {
    "-k": ["life"],
    "-i": [10],
    "-v": ["lazy_ompfor"],
    "-a": ["moultdiehard1398"],
    "-s": [4096],
    "-tw": [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096],
    "-th": [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048],
    "-of": ["data/perf/etape-2/life-heat-opt.csv"],
    "-wt": ["opt"],
    "-ft": [""]
}

# OMP Internal Control Variable
ompICV = {
    "OMP_SCHEDULE": ["static,8"],
    "OMP_PLACES": ["sockets"],
    "OMP_PROC_BIND": ["close"],
    "OMP_NUM_THREADS": [30]
}

nbruns = 1
# Lancement des experiences
execute("./run ", ompICV, easypapOptions, nbruns, verbose=True, easyPath=".")

ompICV = {
    "OMP_NUM_THREADS": [1]
}

del easypapOptions["-tw"]
del easypapOptions["-th"]
easypapOptions["-v"] = ["seq"]
execute("./run ", ompICV, easypapOptions, nbruns, verbose=True, easyPath=".")


print("Recommended plot:")
print(" plots/easyplot.py -if life-heat.csv -v omp_tiled -- col=schedule row=label")
