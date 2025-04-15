#!/usr/bin/env python3

from graphTools import *
from expTools import *
import os

# Dictionnaire avec les options de compilations d'apres commande
options = {}
options["-k"] = ["life"]
options["-i"] = [2000]
options["-v"] = ["lazy_ompfor"]
options["-wt"] = ["avx2", "avx512"]
options["-s"] = [2048]
options["-ft"] = [""]
options["-tw"] = [2048]
options["-th"] = [4]

options["-a"] = ["moultdiehard1398"]

# Pour renseigner l'option '-of' il faut donner le chemin depuis le fichier easypap
options["-of"] = ["./data/perf/life.csv"]


# Dictionnaire avec les options OMP
ompenv = {}
ompenv["OMP_NUM_THREADS"] = [1] + list(range(2, 46, 2))
ompenv["OMP_SCHEDULE"] = [
    "static,8",
    "static,16",
    "static,32",
    "static,64",
    "static,128",
]
ompenv["OMP_PLACES"] = ["threads"]

nbruns = 1
# Lancement des experiences
execute("./run ", ompenv, options, nbruns, verbose=True, easyPath=".")

# Lancement de la version seq avec le nombre de thread impose a 1
del options["-tw"]
del options["-th"]
options["-v"] = ["seq"]
ompenv["OMP_NUM_THREADS"] = [1]
execute("./run", ompenv, options, nbruns, verbose=True, easyPath=".")
print(
    " plots/easyplot.py -if ./data/perf/life.csv -v omp_tiled -- col=schedule row=label"
)
