#!/usr/bin/env python3

from graphTools import *
from expTools import *
import os

# Dictionnaire avec les options de compilations d'apres commande
options = {}
options["-k"] = ["life_gpu"]
options["-i"] = [2470]
options["-v"] = ["ocl", "ocl_lazy", "ocl_2x"]
options["-s"] = [1024]
options["-g"] = [""]
options["-tw"] = [2**x for x in range(0, 5, 1)]
options["-th"] = [2**x for x in range(0, 5, 1)]

options["-a"] = ["moultdiehard2474"]

# Pour renseigner l'option '-of' il faut donner le chemin depuis le fichier easypap
options["-of"] = ["./data/perf/life.csv"]


# Dictionnaire avec les options OMP
ompenv = {}
ompenv["OMP_NUM_THREADS"] = [1]  # + list(range(2, 9, 2))
ompenv["OMP_PLACES"] = ["threads"]

nbruns = 1
# Lancement des experiences
execute("./run ", ompenv, options, nbruns, verbose=True, easyPath=".")

# Lancement de la version seq avec le nombre de thread impose a 1
del options["-tw"]
del options["-th"]
del options["-g"]
options["-v"] = ["seq"]
ompenv["OMP_NUM_THREADS"] = [1]
execute("./run", ompenv, options, nbruns, verbose=True, easyPath=".")
print(
    " plots/easyplot.py -if ./data/perf/life.csv -v omp_tiled -- col=schedule row=label"
)
