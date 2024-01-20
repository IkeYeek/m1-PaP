#!/usr/bin/env python3
from graphTools import *
from expTools import *
import os

# Recommended plot:
# ./plots/easyplot.py --plot catplot -y time -if rotation90.csv  -x variant  -- col=size

easyspap_options = {}
easyspap_options["--kernel"] = ["rotation90"]
easyspap_options["--variant"] = [
    "omp_tiled",
    "omp_cache",
]
easyspap_options["--tile-size"] = [8, 16, 32, 64]
easyspap_options["--counters"] = [""]
easyspap_options["-of"] = ["rotation90.csv"]

omp_icv = {}  # OpenMP Internal Control Variables
omp_icv["OMP_NUM_THREADS"] = [os.cpu_count() // 2]
omp_icv["OMP_SCHEDULE"] = ["static"]

for size in [512, 1024, 2048, 4096]:
    easyspap_options["--iterations"] = [8 * 4096 * 4096 // (size * size)]
    easyspap_options["--size"] = [size]
    execute("./run", omp_icv, easyspap_options, nbruns=4)


print("Recommended plot:")
print("./plots/easyplot.py --plot catplot -y time -if rotation90.csv  -x variant  -- col=size")