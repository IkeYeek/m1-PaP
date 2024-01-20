#!/usr/bin/env python3

from graphTools import *
from expTools import *
import os


# options communes
options = {}
options["--label"] = ["final"]
options["-k"] = ["mandel"]
options["-i"] = [5]
options["-of"] = ["mandel-simple.csv"]
options["-s"] = [512]

ompenv = {}
ompenv["OMP_NUM_THREADS"] = [1] + list(range(4, os.cpu_count() + 1, 4))
nbruns = 5

options["-v"] = ["omp_tiled"]
options["-ts"] = [8, 16, 32]
ompenv["OMP_SCHEDULE"] = ["dynamic"]

# execute('./run ', ompenv, options, nbruns, verbose=False, easyPath=".")

# OMP_LINE static
del options["-ts"]
ompenv["OMP_SCHEDULE"] = ["static"]
options["-tw"] = [512]
options["-th"] = [1]

execute("./run ", ompenv, options, nbruns, verbose=False, easyPath=".")

# OMP_LINE dynamic
ompenv["OMP_SCHEDULE"] = ["dynamic"]
execute("./run ", ompenv, options, nbruns, verbose=False, easyPath=".")

print("Recommended plot:")
print("plots/easyplot.py -if mandel-simple.csv -v omp_tiled")
