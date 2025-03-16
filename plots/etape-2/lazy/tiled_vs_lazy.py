#!/usr/bin/env python3

from graphTools import *
from expTools import *
import os

# Dictionnaire avec les options de compilations d'apres commande
options = {}
options["-k"] = ["life"]
options["-i"] = [5]
options["-v"] = ["lazy", "tiled"]
options["-s"] = [4096]
options["-ts"] = [16]
options["-wt"] = ["opt"]
options["-a"] = ["random", "moultdiehard130", "clown"]

# Pour renseigner l'option '-of' il faut donner le chemin depuis le fichier easypap
options["-of"] = ["./data/perf/etape-2/tiled_vs_lazy.csv"]


# Dictionnaire avec les options OMP
ompenv = {}
ompenv["OMP_NUM_THREADS"] = [1]
ompenv["OMP_PLACES"] = ["threads"]
ompenv["OMP_SCHEDULE"] = ["dynamic", "static,8"]

nbruns = 1
# Lancement des experiences
execute('./run ', ompenv, options, nbruns, verbose=True, easyPath=".")

# Lancement de la version seq avec le nombre de thread impose a 1
options["-v"] = ["tiled"]
ompenv["OMP_NUM_THREADS"] = [1]
#execute('./run', ompenv, options, nbruns, verbose=False, easyPath=".")
print(" plots/easyplot.py -if ./data/perf/life.csv -v omp_tiled -- col=schedule row=label")
