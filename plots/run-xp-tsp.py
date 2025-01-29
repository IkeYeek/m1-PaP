#!/usr/bin/env python3
from expTools import *
import sys

nb_villes = 13 if len(sys.argv) == 1 else sys.argv[1]

easypapOptions = {
    "-of": ["tsp.csv"],
    "-k": ["tsp"],
    "-i": [5],
    "-v": [f"ompfor -a {nb_villes}-2"],
    "--label": ["gnu"],
}

# OMP Internal Control Variable
ompICV = {
    "OMP_SCHEDULE": ["dynamic"],
    "OMP_NUM_THREADS": [os.cpu_count()-2],
}

nbruns = 1
# Lancement des experiences
execute("./run ", ompICV, easypapOptions, nbruns, verbose=False, easyPath=".")


easypapOptions = {
    "-of": ["tsp.csv"],
    "-k": ["tsp"],
    "-i": [5],
    "-v": [f"ompcol2 -a {nb_villes}-2", f"ompcol3 -a {nb_villes}-3", f"ompcol4 -a {nb_villes}-4"],
    "--label": ["gnu"],
}

# OMP Internal Control Variable
ompICV = {
    "OMP_SCHEDULE": ["static,1", "dynamic", "nonmonotonic:dynamic"],
    "OMP_NUM_THREADS": [os.cpu_count()-2],
}

nbruns = 1
# Lancement des experiences
execute("./run ", ompICV, easypapOptions, nbruns, verbose=False, easyPath=".")

ompICV = {"OMP_NUM_THREADS": [os.cpu_count()-2]}
easypapOptions = {
    "-of": ["tsp.csv"],
    "-k": ["tsp"],
    "-i": [5],
    "-v": ["taskdyn", "taskwait", "taskpriv"],
    "-a": [f"{nb_villes}-2", f"{nb_villes}-3", f"{nb_villes}-4"],
    "--label": ["gnu"],
}

execute("./run ", ompICV, easypapOptions, nbruns, verbose=False, easyPath=".")

print("Recommended plot:")
print("./plots/tsp-plot.py -if tsp.csv --noSort -y time -x grain",
" --delete arg -- row=nb_villes col=variant sharey=row sharex=true aspect=1 height=2")