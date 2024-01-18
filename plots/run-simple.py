#!/usr/bin/env python3

from graphTools import *
from expTools import *
import os

# options communes
options = {}
options["--label"] = ["final"]
options["-k"] = ["mandel"]
options["-i"] = [10]
options["-of"] = ["./data/perf/data.csv"]
options["-s"] = [512]

ompenv = {}
ompenv["OMP_NUM_THREADS"] = [1] + list(range(2, 13, 2))

nbruns = 15

options["-v"] = ["omp_tiled"]
options["-ts"] = [8, 16, 32]
ompenv["OMP_SCHEDULE"] = ["dynamic"]

#execute('./run ', ompenv, options, nbruns, verbose=False, easyPath=".")

# OMP_LINE static
del options["-ts"]
ompenv["OMP_SCHEDULE"] = ["static"]
options["-tw"] = [512]
options["-th"] = [1]

execute("./run ", ompenv, options, nbruns, verbose=False, easyPath=".")

# OMP_LINE dynamic
ompenv["OMP_SCHEDULE"] = ["dynamic"]
execute("./run ", ompenv, options, nbruns, verbose=False, easyPath=".")
