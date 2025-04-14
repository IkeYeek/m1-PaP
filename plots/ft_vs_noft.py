#!/usr/bin/env python3

from graphTools import *
from expTools import *
import os

# Dictionnaire avec les options de compilations d'apres commande
options = {}
options["-k"] = ["life"]
options["-i"] = [50]
options["-v"] = ["lazy_ompfor"]
options["-s"] = [4096]
options["-tw"] = [2048]
options["-th"] = [16]
options["-wt"] = ["opt"]
options["-a"] = ["moultdiehard130"]
options["--label"] = ["with-ft"]
options["-ft"] = [""]

# Pour renseigner l'option '-of' il faut donner le chemin depuis le fichier easypap
options["-of"] = ["./data/perf/etape-2/ft_vs_noft.csv"]


# Dictionnaire avec les options OMP
ompenv = {}
ompenv["OMP_NUM_THREADS"] = [30]
ompenv["OMP_SCHEDULE"] = ["static,8"]
ompenv["OMP_PLACES"] = ["sockets"]

nbruns = 1
# Lancement des experiences
execute('./run ', ompenv, options, nbruns, verbose=True, easyPath=".")
del options["-ft"]
options["--label"] = ["without-ft"]
execute('./run ', ompenv, options, nbruns, verbose=True, easyPath=".")


# Lancement de la version seq avec le nombre de thread impose a 1
options["-v"] = ["seq"]
ompenv = {}
ompenv["OMP_NUM_THREADS"] = [1]
execute('./run', ompenv, options, nbruns, verbose=False, easyPath=".")
print(" plots/easyplot.py -if ./data/perf/life.csv -v omp_tiled -- col=schedule row=label")
