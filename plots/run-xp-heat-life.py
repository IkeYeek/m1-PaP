#!/usr/bin/env python3
import os
from expTools import *

# Recommanded plot :
# ./plots/easyplot.py -if heat-life.csv --plottype heatmap -heatx tilew -heaty tileh -v omp_tiled -- row=schedule aspect=1.8

easypapOptions = {
    "-k": ["life"],
    "-i": [5],
    "-v": ["ompfor", "omp_tiled", "omp_task", "omp_workshare"],
    "-s": [256, 512, 1024, 2048],
    "-th": [2**i for i in range(0, 10)] + [64, 128, 192, 256],
    "-tw": [2**i for i in range(0, 10)] + [64, 128, 192, 256],
    "-of": ["heat-life.csv"],
}

# OMP Internal Control Variables
ompICV = {
    "OMP_SCHEDULE": ["dynamic", "static,1", "static", "guided", "auto"],
    "OMP_NUM_THREADS": [os.cpu_count() // 2, os.cpu_count()],
    "OMP_PLACES": ["cores"],
    "OMP_PROC_BIND": ["close", "spread", "master"],
}

# Execute parallel versions with different options
execute("./run ", ompICV, easypapOptions, verbose=True, easyPath=".")

# Lancement de la version séquentielle avec un seul thread
ompICV = {"OMP_NUM_THREADS": [1]}
del easypapOptions["-th"]
del easypapOptions["-tw"]
easypapOptions["-v"] = ["seq"]
execute("./run ", ompICV, easypapOptions, verbose=False, easyPath=".")



print("Recommended plot:")
print(" ./plots/easyplot.py -if heat-life.csv --plottype heatmap", 
      " -heatx tilew -heaty tileh -v ompfor -- row=schedule aspect=1.8")
