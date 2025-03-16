#!/usr/bin/env python3

from graphTools import *
from expTools import *
import os

# Dictionnaire avec les options de compilations d'apres commande
options = {}
options["-k"] = ["life"]
options["-i"] = [5]
options["-v"] = ["lazy_ompfor", "ompfor", "omptaskloop"]
options["-s"] = [8192]
options["-tw"] = [512]
options["-th"] = [16]
options["-wt"] = ["opt"]
options["-a"] = ["random", "moultdiehard130", "clown"]

# Pour renseigner l'option '-of' il faut donner le chemin depuis le fichier easypap
options["-of"] = ["./data/perf/etape-2/lazy_vs_ompfor.csv"]


# Dictionnaire avec les options OMP
ompenv = {}
ompenv["OMP_NUM_THREADS"] = [26]
ompenv["OMP_SCHEDULE"] = ["dynamic", "static,8"]
ompenv["OMP_PLACES"] = ["threads"]

nbruns = 1
# Lancement des experiences
execute('./run ', ompenv, options, nbruns, verbose=True, easyPath=".")

# Lancement de la version seq avec le nombre de thread impose a 1
options["-v"] = ["seq"]
ompenv["OMP_NUM_THREADS"] = [1]
execute('./run', ompenv, options, nbruns, verbose=False, easyPath=".")
print(" plots/easyplot.py -if ./data/perf/life.csv -v omp_tiled -- col=schedule row=label")
