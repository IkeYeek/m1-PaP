#!/usr/bin/env python3

from graphTools import *
from expTools import *
import os

# Dictionnaire avec les options de compilations d'apres commande
options = {}
options["-k"] = ["life"]
options["-i"] = [300]
options["-v"] = ["omptaskloop"]
options["-s"] = [8192]
options["-tw"] = [8192]
options["-th"] = [8]
options["-wt"] = ["default", "opt"]

options["-a"] = ["random"]

# Pour renseigner l'option '-of' il faut donner le chemin depuis le fichier easypap
options["-of"] = ["./data/perf/life-tile-opt.csv"]


# Dictionnaire avec les options OMP
ompenv = {}
ompenv["OMP_NUM_THREADS"] = [20]
ompenv["OMP_PLACES"] = ["sockets"]

nbruns = 1
# Lancement des experiences
execute('./run ', ompenv, options, nbruns, verbose=True, easyPath=".")

# Lancement de la version seq avec le nombre de thread impose a 1
options["-v"] = ["seq"]
ompenv["OMP_NUM_THREADS"] = [1]
execute('./run', ompenv, options, nbruns, verbose=False, easyPath=".")
print(" plots/easyplot.py -if ./data/perf/life.csv -v omp_tiled -- col=schedule row=label")
