#!/usr/bin/env python3

from graphTools import *
from expTools import *
import os

# Dictionnaire avec les options de compilations d'apres commande
options = {}
options["-k"] = ["life"]
options["-i"] = [2000]
options["-v"] = ["lazy"]
options["-wt"] = ["avx2", "avx512", "avx_balanced", "opt"]
options["-s"] = [256]
options["-ft"] = [""]
options["-tw"] = [2**x for x in range(0, 8, 1)]
options["-th"] = [2**x for x in range(0, 8, 1)]

options["-a"] = ["random"]

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
options["-v"] = ["seq"]
ompenv["OMP_NUM_THREADS"] = [1]
execute("./run", ompenv, options, nbruns, verbose=True, easyPath=".")
print(
    " plots/easyplot.py -if ./data/perf/life.csv -v omp_tiled -- col=schedule row=label"
)
